from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PADRAO_NOME = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})[ _](\d+)(?:_(\d+))?$")
SPLITS = ("train", "val", "test")
PASTA_ANOT = "annotations_only_non_and_pollenbee_fixed_final"


def md5(caminho: Path) -> str:
    h = hashlib.md5()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", type=Path, help="pasta PollenbeeDataset")
    ap.add_argument("--saida", type=Path,
                    default=Path("splits/manifest_vnpollenbee.csv"))
    args = ap.parse_args()

    linhas: list[dict] = []
    sem_json: list[str] = []
    sem_imagem: list[str] = []
    nome_estranho: list[str] = []
    tipos: Counter = Counter()
    n_pontos: Counter = Counter()
    larguras: list[float] = []
    alturas: list[float] = []

    for split in SPLITS:
        dir_img = args.raiz / split / "images"
        dir_ann = args.raiz / split / PASTA_ANOT
        if not dir_img.is_dir() or not dir_ann.is_dir():
            raise SystemExit(f"pasta em falta: {dir_img} ou {dir_ann}")

        imagens = {p.stem: p for p in sorted(dir_img.glob("*.jpg"))}
        anotacoes = {p.stem: p for p in sorted(dir_ann.glob("*.json"))
                     if not p.stem.startswith("pollenbee_")}

        sem_json += [f"{split}/{n}" for n in sorted(imagens.keys() - anotacoes.keys())]
        sem_imagem += [f"{split}/{n}" for n in sorted(anotacoes.keys() - imagens.keys())]

        for nome in sorted(imagens.keys() & anotacoes.keys()):
            m = PADRAO_NOME.match(nome)
            if m is None:
                nome_estranho.append(f"{split}/{nome}")
                sessao, indice = "DESCONHECIDA", ""
            else:
                sessao, indice = m.group(1), m.group(2)

            dados = json.loads(anotacoes[nome].read_text(encoding="utf-8"))
            formas = dados.get("shapes", [])
            rotulos = Counter(f["label"] for f in formas)

            for forma in formas:
                tipos[forma.get("shape_type")] += 1
                n_pontos[len(forma["points"])] += 1
                xs = [pt[0] for pt in forma["points"]]
                ys = [pt[1] for pt in forma["points"]]
                larguras.append(max(xs) - min(xs))
                alturas.append(max(ys) - min(ys))

            linhas.append({
                "ficheiro": f"{split}/images/{imagens[nome].name}",
                "md5": md5(imagens[nome]),
                "sessao": sessao,
                "indice": indice,
                "split_publicado": split,
                "largura": dados.get("imageWidth"),
                "altura": dados.get("imageHeight"),
                "n_caixas": len(formas),
                "n_pollen": rotulos.get("pollenbee", 0),
                "tem_imagedata": int(dados.get("imageData") is not None),
            })

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    # ---------------- resumo ----------------
    print(f"\nmanifesto escrito: {args.saida}  ({len(linhas)} linhas)\n")
    print("entradas por conjunto publicado:",
          dict(Counter(l["split_publicado"] for l in linhas)))

    onde = defaultdict(set)
    for l in linhas:
        onde[l["md5"]].add(l["split_publicado"])
    print(f"imagens únicas: {len(onde)}   entradas: {len(linhas)}")
    print("\nonde cada imagem única aparece:")
    for combo, n in Counter("+".join(sorted(s)) for s in onde.values()).most_common():
        print(f"  {combo:<22}{n:>6}")

    por_sessao = defaultdict(Counter)
    caixas = Counter()
    for l in linhas:
        por_sessao[l["sessao"]][l["split_publicado"]] += 1
        caixas[l["sessao"]] += int(l["n_caixas"])
    print(f"\n  {'sessão':<22}{'train':>7}{'val':>6}{'test':>6}{'total':>7}"
          f"{'caixas':>9}{'méd/img':>9}")
    for s in sorted(por_sessao):
        c = por_sessao[s]
        tot = sum(c.values())
        print(f"  {s:<22}{c['train']:>7}{c['val']:>6}{c['test']:>6}{tot:>7}"
              f"{caixas[s]:>9}{caixas[s] / tot:>9.1f}")

    print(f"\ncaixas: {sum(int(l['n_caixas']) for l in linhas)}"
          f"   com pólen: {sum(int(l['n_pollen']) for l in linhas)}")
    print(f"caixa mediana: {statistics.median(larguras):.0f}"
          f" x {statistics.median(alturas):.0f} px")
    print("shape_type:", dict(tipos), "  pontos por forma:", dict(n_pontos))
    print("resoluções:", dict(Counter((l["largura"], l["altura"]) for l in linhas)))
    print("com imageData embutido:", sum(int(l["tem_imagedata"]) for l in linhas))

    for titulo, lista in (("imagens sem JSON", sem_json),
                          ("JSON sem imagem", sem_imagem),
                          ("nomes fora do padrão", nome_estranho)):
        print(f"\n{titulo}: {len(lista)}")
        for n in lista[:10]:
            print("   ", n)


if __name__ == "__main__":
    main()