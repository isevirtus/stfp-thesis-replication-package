"""Cenários hipotéticos nos 16 projetos. Não executa GA nem altera bases."""
import csv
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Usa a cópia corrigida congelada, sem a compressão experimental de AC.
FONTE = Path(r"C:\projetos\STFP\Experimento_COMPARACAO_4_METODOS_16_HISTORICOS_20260902")
DIMENSOES = ["dominio", "ecossistema", "linguagens"]
PRIORIDADES = ["must", "should", "could"]


class Log:
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
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def executar(pasta):
    sys.path.insert(0, str(FONTE))
    from Feature_Extraction.Dimension_Scoring.dimension_scoring import avaliar_todos_as_dimensions
    from Algorithms.Surrogate.team_fit_surrogate import TeamFitSurrogate

    projetos_path = FONTE / "inputs/target_projects.json"
    pesos_path = FONTE / "Feature_Extraction/Dimension_Scoring/pesos_calibrados.json"
    params_path = FONTE / "Calibration_surrogate/BN_Teacher/results_teacher_bn/best_params_teacher_bn.json"
    projetos = json.loads(projetos_path.read_text(encoding="utf-8-sig"))["projects"]
    pesos = json.loads(pesos_path.read_text(encoding="utf-8-sig"))
    sur = TeamFitSurrogate.from_params_file(params_path)
    assert {p["id"] for p in projetos} == {f"P{i}" for i in range(1, 17)}
    projetos.sort(key=lambda p: int(p["id"][1:]))

    print("TESTE: COBERTURA / REDUNDÂNCIA -> DIMENSION SCORING CORRIGIDO -> AT")
    print("Equipes fictícias criadas apenas na memória. Não usa desenvolvedores reais.")
    print("Não executa GA, não usa BN e não aplica compressão de AC.")
    print("AC fica fixo em 0.5 apenas para chamar a interface do surrogate. Não é resultado colaborativo.")
    print("Nenhuma cobertura é um diagnóstico e pode violar o MUST hard externo ao módulo.")
    print("Não confundir extremos destes cenários com ótimo viável na base real.")
    print("Cada integrante hipotético pode cobrir todos os requisitos, sem limite de competências.")
    print("As situações de cálculo são decididas pelos requisitos ativos, não pela cobertura da equipe.")
    print("Parâmetros técnicos:", json.dumps(pesos, ensure_ascii=False, indent=2))
    print("Surrogate:", sur)
    for caminho in [projetos_path, pesos_path, params_path,
                    FONTE / "Feature_Extraction/Dimension_Scoring/dimension_scoring.py",
                    FONTE / "Algorithms/Surrogate/team_fit_surrogate.py"]:
        print("FONTE:", caminho)
        print("SHA256:", hashlib.sha256(caminho.read_bytes()).hexdigest())

    linhas = []
    resumo = []
    detalhes = []
    for projeto in projetos:
        pid = projeto["id"]
        tamanho = int(projeto["team_size"])
        print("\n" + "=" * 78)
        print(f"PROJETO {pid}, tamanho fixo da equipe = {tamanho}")
        for dim in DIMENSOES:
            print(dim, json.dumps(projeto.get(dim, {}), ensure_ascii=False))
        resultados_projeto = []
        cenarios = [("sem_cobertura", 0), ("cobertura_1_integrante", 1),
                    ("cobertura_2_integrantes", 2), ("cobertura_3_integrantes", 3),
                    ("cobertura_todos_integrantes", tamanho)]
        for nome, quantidade in cenarios:
            if quantidade > tamanho:
                print("Cenário ignorado, excede tamanho da equipe:", nome)
                continue
            print(f"\nCENÁRIO {nome}: cada requisito atendido por {quantidade} integrante(s).")
            devs = []
            for indice in range(tamanho):
                dev = {"id": indice + 1}
                for dim in DIMENSOES:
                    requisitos = []
                    if indice < quantidade:
                        for prioridade in PRIORIDADES:
                            requisitos.extend(projeto.get(dim, {}).get(prioridade, []) or [])
                    dev[dim] = requisitos
                devs.append(dev)
            equipe = list(range(1, tamanho + 1))
            scores = avaliar_todos_as_dimensions(equipe, {"developers": devs}, projeto, pesos,
                                                 nota_sem_must=0.0)
            linha = {"projeto": pid, "cenario": nome, "tamanho_equipe": tamanho,
                     "atendimentos_por_requisito": quantidade, "hipotetico": True}
            for dim in DIMENSOES:
                item = scores[dim]
                linha[dim + "_score"] = item["score"]
                linha[dim + "_rotulo"] = item["rotulo"]
                print(f"{dim}: escore={item['score']}, rótulo={item['rotulo']}")
                print("Detalhes retornados pelo módulo:", json.dumps(item["debug"], ensure_ascii=False))
                for prioridade in PRIORIDADES:
                    requisitos = set(str(t).strip().lower() for t in projeto.get(dim, {}).get(prioridade, []) or [])
                    total = len(requisitos)
                    prefixo = dim + "_" + prioridade
                    linha[prefixo + "_total"] = total
                    for limiar, sufixo in [(1, "cobertos"), (2, "red2"), (3, "red3")]:
                        linha[prefixo + "_" + sufixo] = total if quantidade >= limiar else 0
                    print(f"  {prioridade}: total={total}, cobertos={linha[prefixo+'_cobertos']}, "
                          f"red2={linha[prefixo+'_red2']}, red3={linha[prefixo+'_red3']}")
            # Mesmo tratamento das dimensões vazias usado pelo pipeline corrigido.
            dom = 0.5 if scores["dominio"]["score"] is None else scores["dominio"]["score"]
            eco = 0.5 if scores["ecossistema"]["score"] is None else scores["ecossistema"]["score"]
            ling = scores["linguagens"]["score"]
            resultado = sur.evaluate(dom, eco, ling, [0, 0, 1, 0, 0])
            at = resultado["AT_sur"]
            linha["AT"] = at
            linha["MUST_hard_atendido"] = quantidade >= 1 or not any(
                projeto.get(d, {}).get("must", []) for d in DIMENSOES)
            print(f"Entradas efetivas do surrogate: Dom={dom}, Eco={eco}, Ling={ling}")
            print(f"Se Ling=None, surrogate usa neutro={sur.missing_ling}. AT={at:.10f}")
            print("MUST hard atendido neste cenário:", linha["MUST_hard_atendido"])
            assert 0 <= at <= 1
            linhas.append(linha)
            resultados_projeto.append(linha)
            detalhes.append({"projeto": pid, "cenario": nome, "scores": scores,
                             "entrada_surrogate": {"Dom": dom, "Eco": eco, "Ling": ling}, "AT": at})
        ats = [r["AT"] for r in resultados_projeto]
        assert all(b >= a - 1e-10 for a, b in zip(ats, ats[1:])), (pid, ats)
        resumo.append({"projeto": pid, "AT_min_cenarios_testados": min(ats),
                       "AT_max_cenarios_testados": max(ats), "amplitude_testada": max(ats)-min(ats),
                       "AT_cobertura_1": ats[1], "AT_cobertura_2": ats[2], "AT_cobertura_3": ats[3],
                       "AT_todos": ats[-1], "nao_prova_viabilidade_real": True})
        print("RESUMO DO PROJETO:", resumo[-1])

    salvar_csv(pasta / "cenarios_por_projeto.csv", linhas)
    salvar_csv(pasta / "resumo_range_AT.csv", resumo)
    (pasta / "detalhes_dimension_scoring.json").write_text(
        json.dumps(detalhes, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRESUMO DOS 16 PROJETOS")
    for r in resumo:
        print(f"{r['projeto']}: AT mínimo testado={r['AT_min_cenarios_testados']:.8f}, "
              f"AT máximo testado={r['AT_max_cenarios_testados']:.8f}")
    print(f"CONCLUÍDO: {len(linhas)} cenários. Saídas em {pasta}")
    print("Teste de cenários homogêneos, não busca exaustiva de todas as configurações de cobertura.")


if __name__ == "__main__":
    pasta = Path(__file__).resolve().parent / "resultados_dimension" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    stdout, stderr = sys.stdout, sys.stderr
    with (pasta / "execucao.log").open("w", encoding="utf-8") as arquivo:
        sys.stdout, sys.stderr = Log(stdout, arquivo), Log(stderr, arquivo)
        try:
            executar(pasta)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
