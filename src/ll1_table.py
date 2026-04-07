"""
ll1_table.py  -  Tabla de parsing LL(1) y parser predictivo.

Teoria:

  TABLA DE PARSING LL(1):
    Una tabla M[A, a] donde:
      A = no-terminal
      a = terminal (o $)
      M[A, a] = produccion a usar cuando el parser ve 'a' en tope de pila
                y 'a' en la entrada.

    Como se llena:
      Para cada produccion A -> alpha:
        1. Para cada terminal 'a' en FIRST(alpha):
               M[A, a] = A -> alpha
        2. Si epsilon in FIRST(alpha):
               Para cada terminal 'b' en FOLLOW(A):
                   M[A, b] = A -> alpha
               Si $ in FOLLOW(A):
                   M[A, $] = A -> alpha

    La gramatica es LL(1) si y solo si no hay conflictos
    (ninguna celda tiene mas de una produccion).

  CONFLICTOS:
    - FIRST/FIRST: dos producciones A->alpha y A->beta con el
      mismo terminal en FIRST(alpha) ∩ FIRST(beta)
    - FIRST/FOLLOW: cuando epsilon in FIRST(alpha) y un terminal
      esta tanto en FIRST(alpha) como en FOLLOW(A)

  PARSER PREDICTIVO (sin backtracking):
    Usa la tabla para elegir siempre la produccion correcta
    mirando solo el token actual. Es O(n) en lugar de exponencial.

    Algoritmo de la pila:
      Pila: [S, $]   Entrada: w$
      Repetir:
        Sea X el tope de la pila, 'a' el token actual
        Si X == a == $: EXITO
        Si X == a (terminal): pop pila, avanzar entrada
        Si X es NT y M[X,a] = X->Y1Y2..Yk:
            pop X, push Yk...Y2,Y1 (en orden inverso)
        Si M[X,a] = error: RECUPERAR
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from src.cfg_grammar  import Grammar
from src.first_follow import compute_first, compute_follow, first_of_string, EPSILON, EOF_SYM
from src.parse_tree   import ParseNode

# Tipo de la tabla: (NT, terminal) -> lista de producciones (puede haber conflictos)
LLTable = Dict[Tuple[str, str], List[List[str]]]


class LL1Conflict:
    """Representa un conflicto en la tabla LL(1)."""
    def __init__(self, nt: str, terminal: str, prods: list):
        self.nt       = nt
        self.terminal = terminal
        self.prods    = prods  # lista de producciones en conflicto

    def __str__(self):
        alts = "  |  ".join(" ".join(p) if p else "epsilon" for p in self.prods)
        return f"  CONFLICTO M[{self.nt}, '{self.terminal}'] = {{ {alts} }}"


def build_ll1_table(grammar: Grammar) -> Tuple[LLTable, List[LL1Conflict]]:
    """
    Construye la tabla de parsing LL(1).
    Devuelve (tabla, lista_de_conflictos).
    Si la lista esta vacia la gramatica es LL(1).
    """
    first  = compute_first(grammar)
    follow = compute_follow(grammar, first)
    table: LLTable = {}
    conflicts: List[LL1Conflict] = []

    for nt, prods in grammar.productions.items():
        for prod in prods:
            # FIRST de la produccion
            alpha_first = first_of_string(prod, first) if prod else {EPSILON}

            # Regla 1: para cada terminal en FIRST(prod) - {epsilon}
            for terminal in alpha_first - {EPSILON}:
                key = (nt, terminal)
                if key not in table:
                    table[key] = []
                if prod not in table[key]:
                    table[key].append(prod)

            # Regla 2: si epsilon in FIRST(prod) -> usar FOLLOW(A)
            if EPSILON in alpha_first:
                for terminal in follow.get(nt, set()):
                    key = (nt, terminal)
                    if key not in table:
                        table[key] = []
                    if prod not in table[key]:
                        table[key].append(prod)

    # Detectar conflictos
    for (nt, terminal), prods in table.items():
        if len(prods) > 1:
            conflicts.append(LL1Conflict(nt, terminal, prods))

    return table, conflicts


def is_ll1(grammar: Grammar) -> bool:
    """True si la gramatica es LL(1) (sin conflictos)."""
    _, conflicts = build_ll1_table(grammar)
    return len(conflicts) == 0


def print_ll1_table(grammar: Grammar) -> None:
    """Imprime la tabla LL(1) formateada."""
    table, conflicts = build_ll1_table(grammar)
    first  = compute_first(grammar)
    follow = compute_follow(grammar, first)

    # Recopilar todos los terminales relevantes
    all_terms = set()
    for (_, t) in table:
        all_terms.add(t)
    all_terms = sorted(all_terms)

    nts = sorted(grammar.nonterminals)

    print(f"\n{'='*64}")
    print(f"  Tabla de Parsing LL(1)")
    if conflicts:
        print(f"  ATENCION: {len(conflicts)} conflicto(s) — la gramatica NO es LL(1)")
    else:
        print(f"  La gramatica ES LL(1) — sin conflictos")
    print(f"{'='*64}")

    # Encabezado
    col_w = 20
    hdr = f"  {'NT':<20} | " + " | ".join(f"{t:<{col_w}}" for t in all_terms)
    print(hdr)
    print("  " + "-" * len(hdr))

    for nt in nts:
        cells = []
        for t in all_terms:
            prods = table.get((nt, t), [])
            if not prods:
                cells.append(" " * col_w)
            elif len(prods) == 1:
                body = " ".join(prods[0]) if prods[0] else "ε"
                cells.append(f"{nt}->{body:<{col_w-len(nt)-2}}")
            else:
                cells.append(f"{'CONFLICTO':<{col_w}}")
        print(f"  {nt:<20} | " + " | ".join(cells))

    if conflicts:
        print(f"\n  Conflictos detectados:")
        for c in conflicts:
            print(str(c))


def report_ll1(grammar: Grammar) -> str:
    """Reporte compacto sobre si la gramatica es LL(1)."""
    _, conflicts = build_ll1_table(grammar)
    lines = ["Analisis LL(1):"]
    if not conflicts:
        lines.append("  La gramatica ES LL(1): no hay conflictos en la tabla.")
        lines.append("  El parser puede elegir la produccion correcta sin backtracking.")
    else:
        lines.append(f"  La gramatica NO es LL(1): {len(conflicts)} conflicto(s).")
        for c in conflicts:
            lines.append(str(c))
        lines.append("  Aplicar eliminacion de recursividad izquierda y factorizacion")
        lines.append("  puede resolver conflictos y hacer la gramatica LL(1).")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Parser predictivo basado en tabla (sin backtracking)
# ─────────────────────────────────────────────────────────────

class LL1ParseError(Exception):
    pass


class LL1Parser:
    """
    Parser predictivo LL(1) basado en tabla de parsing.
    Mas eficiente que el descendente con backtracking: O(n).
    Solo funciona si la gramatica es LL(1) (sin conflictos).
    """

    def __init__(self, grammar: Grammar,
                 tokens: List[Tuple[str, str]]) -> None:
        self.grammar = grammar
        self.tokens  = [(t, l) for t, l in tokens
                        if t not in ("WS", "WHITESPACE", "NEWLINE")]
        self.table, self.conflicts = build_ll1_table(grammar)
        self.first  = compute_first(grammar)
        self.follow = compute_follow(grammar, self.first)

    def is_ll1(self) -> bool:
        return len(self.conflicts) == 0

    def parse(self) -> ParseNode:
        """
        Algoritmo de parsing predictivo por tabla.
        Usa una pila explicita en lugar de recursion.
        """
        if not self.is_ll1():
            raise LL1ParseError(
                f"La gramatica tiene {len(self.conflicts)} conflicto(s) LL(1). "
                f"No se puede usar el parser predictivo.\n"
                f"Usa el parser descendente con backtracking (sin --ll1)."
            )

        # Tokens + centinela EOF
        input_tokens = self.tokens + [(EOF_SYM, EOF_SYM)]
        pos = 0

        # Pila de (simbolo, nodo_del_arbol)
        root = ParseNode(self.grammar.start)
        stack: List[Tuple[str, ParseNode]] = [
            (EOF_SYM, None),
            (self.grammar.start, root),
        ]

        while stack:
            top_sym, top_node = stack[-1]
            cur_type, cur_lex = input_tokens[pos]

            if top_sym == EOF_SYM:
                if cur_type == EOF_SYM:
                    break   # EXITO
                else:
                    raise LL1ParseError(
                        f"Entrada no consumida: {input_tokens[pos:]}"
                    )

            if top_sym in self.grammar.terminals:
                # Terminal en tope de pila
                if self._match(top_sym, cur_type, cur_lex):
                    top_node.lexeme = cur_lex
                    stack.pop()
                    pos += 1
                else:
                    raise LL1ParseError(
                        f"Error: se esperaba '{top_sym}' pero se encontro "
                        f"'{cur_lex}' ({cur_type}) en posicion {pos}"
                    )
            else:
                # NT en tope de pila: consultar tabla
                # Intentar con tipo y luego con lexema
                prod = self.table.get((top_sym, cur_type)) or \
                       self.table.get((top_sym, cur_lex))

                if not prod:
                    raise LL1ParseError(
                        f"Error: no hay produccion para M[{top_sym}, '{cur_type}']. "
                        f"Token: '{cur_lex}' en posicion {pos}"
                    )

                production = prod[0]  # LL(1): siempre una sola
                stack.pop()

                # Crear hijos del nodo
                child_nodes = []
                for sym in production:
                    child = ParseNode(sym)
                    top_node.children.append(child)
                    child_nodes.append((sym, child))

                # Apilar en orden inverso
                for sym, node in reversed(child_nodes):
                    stack.append((sym, node))

        return root

    @staticmethod
    def _match(expected: str, tok_type: str, tok_lexeme: str) -> bool:
        return expected == tok_type or expected == tok_lexeme
