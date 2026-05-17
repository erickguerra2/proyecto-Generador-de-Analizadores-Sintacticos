#!/usr/bin/env python3
"""Pipeline principal: .yal + .yapar -> tokens -> LL(1) | SLR(1) | LALR."""

import sys, os, argparse, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

from src.cfg_grammar         import Grammar
from src.ambiguity           import report_fix_ambiguity
from src.error_recovery      import report_fix_production_issues
from src.first_follow        import report_first_follow
from src.yapar_parser        import parse_yapar, YAParError
from src.lexer.afd_to_cfg    import print_afd_cfg_conversion, load_afd_from_lexer

from src.ll1.left_recursion  import has_left_recursion, eliminate_left_recursion, report_left_recursion
from src.ll1.factorization   import needs_factorization, left_factor
from src.ll1.ll1_table       import build_ll1_table, print_ll1_table, LL1Parser, LL1ParseError

from src.lr.lr0              import build_lr0, report_lr0
from src.slr1.slr1           import build_slr1_table, SLR1Parser, SLR1ParseError, report_slr1


def generate_lexer_from_yal(yal_path: str) -> str:
    import subprocess
    out_path = os.path.join("output", os.path.basename(yal_path).replace(".yal", "_generated.py"))
    os.makedirs("output", exist_ok=True)
    r = subprocess.run([sys.executable, "src/lexer/generator.py", yal_path, "-o", out_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    print(f"Lexer generado: {out_path}")
    return out_path


def load_lexer(lexer_path: str):
    abs_path = os.path.abspath(lexer_path)
    if not os.path.exists(abs_path):
        print(f"[ERROR] Lexer no encontrado: {abs_path}"); sys.exit(1)
    spec = importlib.util.spec_from_file_location("_lexer_mod", abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokenize_source(text: str, lexer_mod) -> list:
    try:
        return lexer_mod.yylex(text)
    except Exception as e:
        print(f"[ERROR LEXICO] {e}"); sys.exit(1)


def load_grammar(args) -> tuple:
    ignored_tokens = set()
    if args.yapar:
        if not os.path.exists(args.yapar):
            print(f"[ERROR] No encontrado: {args.yapar}"); sys.exit(1)
        try:
            grammar, ignored_tokens = parse_yapar(args.yapar)
            print(f"Gramatica cargada: {args.yapar}")
            print(f"Tokens ignorados : {', '.join(sorted(ignored_tokens)) or 'ninguno'}")
        except YAParError as e:
            print(f"[ERROR YAPAR] {e}"); sys.exit(1)
    else:
        if not os.path.exists(args.grammar):
            print(f"[ERROR] No encontrado: {args.grammar}"); sys.exit(1)
        grammar = Grammar.from_file(args.grammar)
        print(f"Gramatica cargada: {args.grammar}")
    return grammar, ignored_tokens


def preprocess(grammar: Grammar) -> tuple:
    """Preprocesamiento comun a todos los parsers."""
    grammar, prod_report, prod_applied = report_fix_production_issues(grammar)
    print(prod_report)

    grammar, amb_report, amb_applied = report_fix_ambiguity(grammar)
    print(amb_report)

    return grammar, prod_applied | amb_applied


def run_ll1(grammar: Grammar, tokens: list, applied: set, show_grammar: bool) -> None:
    print(report_left_recursion(grammar))
    if "left_recursion_eliminated" not in applied and has_left_recursion(grammar):
        grammar = eliminate_left_recursion(grammar)
        print("Recursividad izquierda eliminada")

    if "factorized" not in applied and needs_factorization(grammar):
        grammar = left_factor(grammar)
        print("Factorizacion aplicada")

    _, conflicts = build_ll1_table(grammar)
    if conflicts:
        print(f"[ERROR] La gramatica no es LL(1): {len(conflicts)} conflicto(s)")
        for c in conflicts: print(f"  {c}")
        sys.exit(1)
    print("Gramatica LL(1) verificada")

    if show_grammar:
        for nt, prods in grammar.productions.items():
            for p in prods:
                print(f"  {nt} -> {' '.join(p) if p else 'epsilon'}")

    print(report_first_follow(grammar))
    print_ll1_table(grammar)

    try:
        parser = LL1Parser(grammar, tokens)
        parser.parse()
    except LL1ParseError as e:
        print(f"[ERROR SINTACTICO] {e}"); sys.exit(1)

    if parser.recovery_log:
        print(parser.recovery_report())
    print("Cadena aceptada (LL(1))")


def run_slr1(grammar: Grammar, tokens: list, show_grammar: bool) -> None:
    states, aug_start = build_lr0(grammar)
    print(report_lr0(states))

    print(report_slr1(grammar))

    table, _, _ = build_slr1_table(grammar)
    if table.has_conflicts():
        print(f"[ADVERTENCIA] {len(table.conflicts)} conflicto(s) SLR(1):")
        for c in table.conflicts: print(str(c))

    table.print_table(
        terminals=sorted(grammar.terminals | {"$"}),
        nonterminals=sorted(grammar.nonterminals)
    )

    try:
        parser = SLR1Parser(grammar, tokens)
        parser.parse()
    except SLR1ParseError as e:
        print(f"[ERROR SINTACTICO] {e}"); sys.exit(1)

    if parser.recovery_log:
        print(parser.recovery_report())
    print("Cadena aceptada (SLR(1))")


def main():
    ap = argparse.ArgumentParser(description="Proyecto 2 - Generador de Analizadores Sintacticos")
    grm_group = ap.add_mutually_exclusive_group(required=True)
    grm_group.add_argument("grammar", nargs="?", help="Archivo .grm")
    grm_group.add_argument("--yapar", "-p")

    lex_group = ap.add_mutually_exclusive_group(required=True)
    lex_group.add_argument("--yal",   "-y")
    lex_group.add_argument("--lexer", "-l")

    txt_group = ap.add_mutually_exclusive_group(required=True)
    txt_group.add_argument("--text", "-t")
    txt_group.add_argument("--file", "-f")

    ap.add_argument("--parser",       "-m", choices=["ll1", "slr1", "lalr"], default="slr1")
    ap.add_argument("--show-grammar", action="store_true")
    ap.add_argument("--afd-to-cfg",   action="store_true")
    args = ap.parse_args()

    # Lexer
    lexer_path = generate_lexer_from_yal(args.yal) if args.yal else args.lexer
    lexer_mod  = load_lexer(lexer_path)

    if args.afd_to_cfg:
        trans, acc, start = load_afd_from_lexer(lexer_mod)
        print_afd_cfg_conversion(trans, acc, start)

    # Gramatica
    grammar, ignored_tokens = load_grammar(args)

    # Preprocesamiento comun
    grammar, applied = preprocess(grammar)

    # Tokenizar
    source     = args.text if args.text else open(args.file, encoding="utf-8").read()
    raw_tokens = tokenize_source(source, lexer_mod)
    skip       = {"WS", "WHITESPACE", "NEWLINE"} | ignored_tokens
    tokens     = [(t, l) for t, l in raw_tokens if t not in skip]
    print(f"Tokens reconocidos: {len(tokens)}")
    for tok, lex in tokens:
        print(f"  {tok:<20} '{lex}'")

    # Parser seleccionado
    if args.parser == "ll1":
        run_ll1(grammar, tokens, applied, args.show_grammar)
    elif args.parser == "slr1":
        run_slr1(grammar, tokens, args.show_grammar)
    elif args.parser == "lalr":
        print("[ERROR] LALR aun no implementado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
