#!/usr/bin/env python3
"""
parser_main.py  -  Proyecto 2: Parser Sintactico con Arbol de Derivacion
=========================================================================

Teoria implementada (PDF + PPT Analisis Sintactico):

  [Diap. 6]  Jerarquia de Chomsky (Tipo 0-3)
  [Diap. 7]  Tipos de parser: descendente / ascendente / universal
  [Diap. 8]  Precedencia y asociatividad
  [Diap. 9]  CFG = (Sigma, N, P, S)
  [Diap. 11-13] Derivaciones por la izquierda
  [Diap. 14-17] Arboles de analisis sintactico
  [Diap. 19-20] Ambiguedad y deteccion
  [Diap. 21]    Eliminar ambiguedad con jerarquia de precedencia
  [Diap. 22-23] Recursividad izquierda y eliminacion
  [Diap. 24-25] Factorizacion por la izquierda
  [Teoria LL]   Conjuntos FIRST y FOLLOW
  [Teoria LL]   Tabla de parsing LL(1) y parser predictivo
  [PDF p.1]     Tipos de errores: Lexical / Syntax / Semantic / Logic
  [PDF p.1]     Recuperacion: panic-mode / phrase-level /
                              production-level / global (min edicion)
  [CNF]         Forma Normal de Chomsky (START/DEL/UNIT/TERM/BIN)

USO:
  # Desde .yal (conecta con Proyecto 1) + texto
  python parser_main.py examples/grammar_simple.grm \\
         --yal examples/ejemplo.yal --text "x = c + 1"

  # Desde archivo
  python parser_main.py examples/grammar_ejemplo2.grm \\
         --yal examples/ejemplo2.yal --file test.txt

  # Mostrar todo: FIRST/FOLLOW + tabla LL(1) + arbol + derivacion
  python parser_main.py examples/grammar_ejemplo2.grm \\
         --yal examples/ejemplo2.yal --file test.txt \\
         --show-grammar --derivation --viz

  # Usar parser predictivo LL(1) en lugar del descendente con backtracking
  python parser_main.py examples/grammar_ejemplo2.grm \\
         --yal examples/ejemplo2.yal --file test.txt --ll1

  # C completo
  python parser_main.py examples/grammar_c.grm \\
         --yal examples/expresiones.yal --file test.c
"""

import sys, os, argparse, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

from src.cfg_grammar    import Grammar
from src.left_recursion import has_left_recursion,  eliminate_left_recursion
from src.right_recursion import report_recursion
from src.factorization  import needs_factorization, left_factor
from src.chomsky        import to_cnf, is_cnf
from src.ambiguity      import report_ambiguity, is_ambiguous
from src.precedence     import analyze_precedence
from src.first_follow   import compute_first, compute_follow, report_first_follow
from src.ll1_table      import build_ll1_table, print_ll1_table, report_ll1, \
                                LL1Parser, LL1ParseError
from src.syntax_parser  import SyntaxParser, ParseError
from src.error_recovery import (panic_mode_recovery, phrase_level_recovery,
                                 production_level_report, global_min_edit_distance,
                                 format_errors)
from src.tree_viz       import print_ascii_tree, render_graphviz, print_derivation


# ─────────────────────────────────────────────────────────────
# Pipeline: .yal -> lexer  (Proyecto 1)
# ─────────────────────────────────────────────────────────────

def generate_lexer_from_yal(yal_path: str) -> str:
    import subprocess
    out_path = os.path.join("output",
               os.path.basename(yal_path).replace(".yal", "_generated.py"))
    os.makedirs("output", exist_ok=True)
    print(f"\n[PIPELINE] Ejecutando Proyecto 1: {yal_path}")
    r = subprocess.run(
        [sys.executable, "src/generator.py", yal_path, "-o", out_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    for line in r.stdout.splitlines():
        if any(k in line for k in ["Reglas:", "estados", "Listo", "Generando"]):
            print(f"  {line.strip()}")
    print(f"[PIPELINE] Lexer listo: {out_path}")
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


# ─────────────────────────────────────────────────────────────
# Helpers de presentacion
# ─────────────────────────────────────────────────────────────

def print_grammar(grammar: Grammar, title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    print(f"  Simbolo inicial (S) : {grammar.start}")
    print(f"  No-terminales   (N) : {', '.join(sorted(grammar.nonterminals))}")
    print(f"  Terminales      (Σ) : {', '.join(sorted(grammar.terminals))}")
    print(f"  CNF                 : {'Si' if is_cnf(grammar) else 'No'}")
    print(f"  LL(1)               : {'Si' if not build_ll1_table(grammar)[1] else 'No (conflictos)'}")
    print()
    for nt, prods in grammar.productions.items():
        for p in prods:
            body = " ".join(p) if p else "ε"
            print(f"  {nt:<24} -> {body}")


def print_tokens(tokens: list) -> None:
    visible = [(t, l) for t, l in tokens if t not in ("WS","WHITESPACE","NEWLINE")]
    skipped = len(tokens) - len(visible)
    print(f"\n{'='*64}")
    print(f"  Tokens  ({len(tokens)} total — {skipped} espacios/saltos filtrados)")
    print(f"{'='*64}")
    for i, (tok, lex) in enumerate(visible):
        print(f"  [{i:3d}]  {tok:<26}  '{lex}'")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Proyecto 2 - Parser sintactico con arbol de derivacion",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("grammar")

    lex_group = ap.add_mutually_exclusive_group(required=True)
    lex_group.add_argument("--yal",   "-y")
    lex_group.add_argument("--lexer", "-l")

    txt_group = ap.add_mutually_exclusive_group(required=True)
    txt_group.add_argument("--text",  "-t")
    txt_group.add_argument("--file",  "-f")

    ap.add_argument("--viz",          action="store_true")
    ap.add_argument("--out",          default="output/parse_tree")
    ap.add_argument("--derivation",   "-d", action="store_true")
    ap.add_argument("--show-grammar", action="store_true")
    ap.add_argument("--cnf",          action="store_true")
    ap.add_argument("--ll1",          action="store_true",
        help="Usar parser predictivo LL(1) (sin backtracking)")
    ap.add_argument("--recovery",     default="panic",
        choices=["panic","phrase","none"])
    args = ap.parse_args()

    # ─── Header ───────────────────────────────────────────────
    print("\n" + "="*64)
    print("  PROYECTO 2 — Parser Sintactico con Arbol de Derivacion")
    print("  Jerarquia Chomsky : Tipo 2 — Lenguajes libres de contexto")
    print("  Parser            : Descendente top-down (LL)")
    print("="*64)

    # ─── Lexer ────────────────────────────────────────────────
    lexer_path = generate_lexer_from_yal(args.yal) if args.yal else args.lexer
    lexer_mod  = load_lexer(lexer_path)
    accept     = getattr(lexer_mod, "ACCEPT", {})
    tok_types  = sorted(set(accept.values()))
    print(f"\n[LEXER] Tokens reconocidos por el lexer generado (Proyecto 1):")
    for i in range(0, len(tok_types), 6):
        print("  " + "  ".join(tok_types[i:i+6]))

    # ─── Gramatica ────────────────────────────────────────────
    if not os.path.exists(args.grammar):
        print(f"[ERROR] Gramatica no encontrada: {args.grammar}"); sys.exit(1)
    grammar = Grammar.from_file(args.grammar)

    if args.show_grammar:
        print_grammar(grammar, "Gramatica original — G = (Σ, N, P, S)")

    # ─── Analisis teorico de la gramatica ─────────────────────

    # Tipos de errores (PDF p.1)
    print(f"\n{'='*64}")
    print("  Tipos de Errores (PDF p.1)")
    print(f"{'='*64}")
    print("  Lexical  : errores de ortografia (manejado por Proyecto 1 - lexer)")
    print("  Syntax   : oracion mal formada   (manejado por ESTE modulo)")
    print("  Semantic : errores de tipos       (fuera de alcance Proyecto 2)")
    print("  Logic    : errores de logica      (fuera de alcance Proyecto 2)")

    # Ambiguedad (diap. 19-20)
    print("\n" + report_ambiguity(grammar))

    # Precedencia y asociatividad (diap. 8)
    print("\n" + analyze_precedence(grammar))

    # Recursividad (diap. 22)
    print("\n" + report_recursion(grammar))

    # ─── Transformaciones de la gramatica ─────────────────────

    # Eliminar recursividad izquierda (diap. 22)
    if has_left_recursion(grammar):
        print("\n[TRANSF] Recursividad IZQUIERDA -> eliminando (diap. 22)...")
        grammar = eliminate_left_recursion(grammar)
        if args.show_grammar:
            print_grammar(grammar, "Sin recursividad izquierda")

    # Factorizacion (diap. 24-25)
    if needs_factorization(grammar):
        print("\n[TRANSF] Prefijos comunes -> factorizando (diap. 24-25)...")
        grammar = left_factor(grammar)
        if args.show_grammar:
            print_grammar(grammar, "Gramatica factorizada")

    # CNF opcional
    if args.cnf:
        if not is_cnf(grammar):
            print("\n[TRANSF] Convirtiendo a CNF (START->DEL->UNIT->TERM->BIN)...")
            grammar = to_cnf(grammar, verbose=args.show_grammar)

    # ─── FIRST / FOLLOW ───────────────────────────────────────
    print("\n" + report_first_follow(grammar))

    # ─── Tabla LL(1) ──────────────────────────────────────────
    print("\n" + report_ll1(grammar))
    if args.show_grammar:
        print_ll1_table(grammar)

    # Production-level (PDF p.1)
    print("\n" + production_level_report(grammar))

    # Gramatica final
    print_grammar(grammar, "Gramatica final usada por el parser")

    # ─── Tokenizar ────────────────────────────────────────────
    source = args.text if args.text else open(args.file, encoding="utf-8").read()
    print(f"\n[INPUT] {repr(source[:100])}{'...' if len(source)>100 else ''}")
    raw_tokens = tokenize_source(source, lexer_mod)
    print_tokens(raw_tokens)

    # Global min-edit (PDF p.1)
    print("\n" + global_min_edit_distance(raw_tokens, grammar.terminals))

    # ─── Parsear ──────────────────────────────────────────────
    tree   = None
    errors = []

    if args.ll1 and not build_ll1_table(grammar)[1]:
        # Parser predictivo LL(1) sin backtracking
        print(f"\n[PARSER] Usando parser predictivo LL(1) (sin backtracking)")
        try:
            ll1 = LL1Parser(grammar, raw_tokens)
            tree = ll1.parse()
        except LL1ParseError as e:
            print(f"\n[ERROR LL(1)] {e}")
            sys.exit(1)
    else:
        if args.ll1:
            print("\n[INFO] La gramatica tiene conflictos LL(1) -> usando parser con backtracking")
        print(f"\n[PARSER] Usando parser descendente con retroceso")
        print(f"[PARSER] Recuperacion de errores: {args.recovery}-mode (PDF p.1)")
        parser = SyntaxParser(grammar, raw_tokens, recovery=args.recovery)
        tree, errors = parser.parse_with_recovery()

    # Errores recuperados
    if errors:
        print(f"\n{'='*64}")
        print(f"  Recuperacion de Errores ({args.recovery}-mode) — PDF p.1")
        print(f"{'='*64}")
        for e in errors:
            print(str(e))

    if tree is None:
        print("\n[FATAL] No se pudo construir el arbol sintactico.")
        sys.exit(1)

    # ─── Arbol ────────────────────────────────────────────────
    print_ascii_tree(tree, title="Arbol de Derivacion Sintactica (diap. 14-17)")

    # Derivacion por la izquierda (diap. 11-13)
    if args.derivation:
        print_derivation(tree)

    # PNG
    if args.viz:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        render_graphviz(tree, output_path=args.out, fmt="png")
        print(f"\n[OUTPUT] Imagen guardada: {args.out}.png")

    # ─── Resumen ──────────────────────────────────────────────
    visible = [t for t in raw_tokens if t[0] not in ("WS","WHITESPACE","NEWLINE")]
    _, ll1_conflicts = build_ll1_table(grammar)
    print(f"\n{'='*64}")
    print("  Resumen del Analisis")
    print(f"{'='*64}")
    print(f"  Tokens procesados    : {len(visible)}")
    print(f"  Errores recuperados  : {len(errors)}")
    print(f"  Arbol generado       : {'Si' if tree else 'No'}")
    print(f"  Gramatica ambigua    : {'Si (ver advertencias)' if is_ambiguous(grammar) else 'No detectada'}")
    print(f"  Gramatica LL(1)      : {'Si' if not ll1_conflicts else 'No (' + str(len(ll1_conflicts)) + ' conflictos)'}")
    print(f"  Gramatica en CNF     : {'Si' if is_cnf(grammar) else 'No'}")
    print(f"  Parser usado         : {'LL(1) predictivo' if args.ll1 and not ll1_conflicts else 'Descendente con backtracking'}")


if __name__ == "__main__":
    main()
