"""Valida a compressao global de AC e compara escalas genericas de AT e AC.

Este script nao executa o GA, nao modifica o avaliador e nao usa projetos reais.
Ele importa o surrogate principal e confere seus resultados com a formula esperada.
"""

import csv
import hashlib
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path


RAIZ_STFP = Path(r"C:\projetos\STFP")
ARQUIVO_SURROGATE = (
    RAIZ_STFP / "Algorithms" / "Surrogate" / "team_fit_surrogate.py"
)
CENTRO_COMPRESSAO = 0.5
FATOR_ESPERADO = 0.5
TOLERANCIA = 1e-12

ESTADOS_AC = [
    ("VL", 0.1, [1, 0, 0, 0, 0]),
    ("L", 0.3, [0, 1, 0, 0, 0]),
    ("M", 0.5, [0, 0, 1, 0, 0]),
    ("H", 0.7, [0, 0, 0, 1, 0]),
    ("VH", 0.9, [0, 0, 0, 0, 1]),
]


class LogDuplo:
    """Escreve simultaneamente no terminal e no arquivo de log."""

    def __init__(self, terminal, arquivo):
        self.terminal = terminal
        self.arquivo = arquivo

    def write(self, texto):
        self.terminal.write(texto)
        self.arquivo.write(texto)
        self.arquivo.flush()

    def flush(self):
        self.terminal.flush()
        self.arquivo.flush()


def salvar_csv(caminho, linhas):
    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


def compressao_esperada(ac_original, fator=FATOR_ESPERADO):
    return CENTRO_COMPRESSAO + fator * (ac_original - CENTRO_COMPRESSAO)


def executar(pasta_resultados):
    sys.path.insert(0, str(RAIZ_STFP))

    from Algorithms.Surrogate.team_fit_surrogate import (
        TeamFitSurrogate,
        at_surrogate,
    )

    surrogate = TeamFitSurrogate.from_default()

    print("EXPERIMENTO DE SENSIBILIDADE DAS ESCALAS AT E AC")
    print("Nao executa GA, nao usa projetos reais e nao modifica o avaliador.")
    print("Surrogate importado de:", ARQUIVO_SURROGATE)
    print("SHA256 do surrogate:", hashlib.sha256(
        ARQUIVO_SURROGATE.read_bytes()
    ).hexdigest())
    print("Configuracao carregada:", surrogate)
    print()

    assert math.isclose(
        surrogate.ac_compression,
        FATOR_ESPERADO,
        abs_tol=TOLERANCIA,
    ), (
        "O surrogate nao esta usando o fator esperado. "
        f"Esperado={FATOR_ESPERADO}; obtido={surrogate.ac_compression}"
    )
    assert surrogate.w_min == 5
    assert surrogate.w_max == 1

    print("1. VALIDACAO DA COMPRESSAO NO SURROGATE REAL")
    print("Formula esperada: AC' = 0.5 + 0.5 * (AC - 0.5)")
    print("AT foi fixado em 0.5 apenas para completar a chamada do avaliador.")
    print()

    linhas_ac = []
    for estado, valor_esperado_original, distribuicao in ESTADOS_AC:
        resultado = surrogate.evaluate(
            Dom=0.5,
            Eco=0.5,
            Ling=0.5,
            ac_dist=distribuicao,
        )

        ac_original = resultado["AC_original"]
        ac_comprimido = resultado["AC_transformado"]
        esperado = compressao_esperada(ac_original)

        original_correto = math.isclose(
            ac_original,
            valor_esperado_original,
            abs_tol=TOLERANCIA,
        )
        compressao_correta = math.isclose(
            ac_comprimido,
            esperado,
            abs_tol=TOLERANCIA,
        )

        assert original_correto
        assert compressao_correta

        linha = {
            "estado": estado,
            "AC_original": ac_original,
            "AC_comprimido": ac_comprimido,
            "AC_comprimido_esperado": esperado,
            "distancia_original_ate_0_5": abs(ac_original - 0.5),
            "distancia_comprimida_ate_0_5": abs(ac_comprimido - 0.5),
            "AT_fixo_no_teste": resultado["AT_sur"],
            "AE_MIXMINMAX": resultado["AE_sur"],
            "teste": "OK",
        }
        linhas_ac.append(linha)

        print(
            f"{estado}: AC original={ac_original:.4f} | "
            f"AC comprimido={ac_comprimido:.4f} | "
            f"esperado={esperado:.4f} | OK"
        )

    valores_originais = [linha["AC_original"] for linha in linhas_ac]
    valores_comprimidos = [linha["AC_comprimido"] for linha in linhas_ac]

    assert all(
        atual < seguinte
        for atual, seguinte in zip(
            valores_comprimidos,
            valores_comprimidos[1:],
        )
    ), "A compressao nao preservou a ordem dos estados."

    amplitude_original = max(valores_originais) - min(valores_originais)
    amplitude_comprimida = max(valores_comprimidos) - min(valores_comprimidos)
    assert math.isclose(
        amplitude_comprimida,
        FATOR_ESPERADO * amplitude_original,
        abs_tol=TOLERANCIA,
    ), "A amplitude de AC nao foi reduzida pelo fator esperado."

    print()
    print("2. PROPRIEDADES CONFERIDAS")
    print("Ordem VL < L < M < H < VH preservada: OK")
    print("Ponto central 0.5 preservado: OK")
    print(
        f"Amplitude operacional original: {amplitude_original:.4f} "
        f"({min(valores_originais):.2f} ate {max(valores_originais):.2f})"
    )
    print(
        f"Amplitude operacional comprimida: {amplitude_comprimida:.4f} "
        f"({min(valores_comprimidos):.2f} ate {max(valores_comprimidos):.2f})"
    )
    print("Reducao da amplitude pela metade: OK")

    at_min_teorico = at_surrogate(0.0, 0.0, 0.0)
    at_max_teorico = at_surrogate(1.0, 1.0, 1.0)
    at_min_centroides = at_surrogate(0.1, 0.1, 0.1)
    at_max_centroides = at_surrogate(0.9, 0.9, 0.9)

    escalas = [
        {
            "indicador": "AT_teorico_com_entradas_0_a_1",
            "minimo": at_min_teorico,
            "maximo": at_max_teorico,
            "amplitude": at_max_teorico - at_min_teorico,
            "observacao": "Limite matematico generico; nao e por projeto.",
        },
        {
            "indicador": "AT_com_tres_entradas_nos_centroides",
            "minimo": at_min_centroides,
            "maximo": at_max_centroides,
            "amplitude": at_max_centroides - at_min_centroides,
            "observacao": "Todos Dom, Eco e Ling em VL ou todos em VH.",
        },
        {
            "indicador": "AC_original_operacional",
            "minimo": min(valores_originais),
            "maximo": max(valores_originais),
            "amplitude": amplitude_original,
            "observacao": "Centróides VL a VH usados pelo surrogate.",
        },
        {
            "indicador": "AC_comprimido_operacional",
            "minimo": min(valores_comprimidos),
            "maximo": max(valores_comprimidos),
            "amplitude": amplitude_comprimida,
            "observacao": "Compressao global com centro 0.5 e fator 0.5.",
        },
        {
            "indicador": "AC_comprimido_teorico_para_entrada_0_a_1",
            "minimo": compressao_esperada(0.0),
            "maximo": compressao_esperada(1.0),
            "amplitude": compressao_esperada(1.0) - compressao_esperada(0.0),
            "observacao": "Limite da transformacao; AC operacional usa 0.1 a 0.9.",
        },
    ]

    print()
    print("3. RESUMO DAS ESCALAS GENERICAS")
    for linha in escalas:
        print(
            f"{linha['indicador']}: "
            f"min={linha['minimo']:.4f} | "
            f"max={linha['maximo']:.4f} | "
            f"amplitude={linha['amplitude']:.4f}"
        )

    base = 0.5
    passo = 0.1
    at_base = at_surrogate(base, base, base)
    sensibilidade = [
        {
            "alteracao": "Dominio: 0.5 para 0.6",
            "valor_antes": at_base,
            "valor_depois": at_surrogate(base + passo, base, base),
            "variacao_saida": at_surrogate(base + passo, base, base) - at_base,
            "formula": "peso 3/9",
        },
        {
            "alteracao": "Ecossistema: 0.5 para 0.6",
            "valor_antes": at_base,
            "valor_depois": at_surrogate(base, base + passo, base),
            "variacao_saida": at_surrogate(base, base + passo, base) - at_base,
            "formula": "peso 1/9",
        },
        {
            "alteracao": "Linguagem: 0.5 para 0.6",
            "valor_antes": at_base,
            "valor_depois": at_surrogate(base, base, base + passo),
            "variacao_saida": at_surrogate(base, base, base + passo) - at_base,
            "formula": "peso 5/9",
        },
        {
            "alteracao": "AC original: 0.5 para 0.6",
            "valor_antes": compressao_esperada(base),
            "valor_depois": compressao_esperada(base + passo),
            "variacao_saida": (
                compressao_esperada(base + passo)
                - compressao_esperada(base)
            ),
            "formula": "fator de compressao 0.5",
        },
    ]

    print()
    print("4. SENSIBILIDADE LOCAL PARA AUMENTO DE 0.1 NA ENTRADA")
    for linha in sensibilidade:
        print(
            f"{linha['alteracao']}: variacao={linha['variacao_saida']:.6f}"
        )

    salvar_csv(
        pasta_resultados / "validacao_compressao_ac.csv",
        linhas_ac,
    )
    salvar_csv(
        pasta_resultados / "resumo_escalas_genericas.csv",
        escalas,
    )
    salvar_csv(
        pasta_resultados / "sensibilidade_local.csv",
        sensibilidade,
    )

    print()
    print("CONCLUIDO: todos os testes passaram.")
    print("Resultados salvos em:", pasta_resultados)
    print()
    print("LIMITACAO:")
    print(
        "Este teste valida a transformacao e as escalas genericas. "
        "Ele nao mede os limites estruturais de AT dos 16 projetos."
    )
    print(
        "Para limites por projeto, use o experimento "
        "Experimento_TESTE_RANGE_DIMENSION_SCORING_20260903."
    )


if __name__ == "__main__":
    pasta_raiz = Path(__file__).resolve().parent
    pasta_resultados = (
        pasta_raiz
        / "resultados"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    pasta_resultados.mkdir(parents=True, exist_ok=False)

    terminal_saida = sys.stdout
    terminal_erro = sys.stderr

    with (pasta_resultados / "execucao.log").open(
        "w",
        encoding="utf-8",
    ) as arquivo_log:
        sys.stdout = LogDuplo(terminal_saida, arquivo_log)
        sys.stderr = LogDuplo(terminal_erro, arquivo_log)

        try:
            executar(pasta_resultados)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        finally:
            sys.stdout = terminal_saida
            sys.stderr = terminal_erro
