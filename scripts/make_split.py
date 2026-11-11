#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import json

PASTA_ANOT = "annotations_only_non_and_pollenbee_fixed_final"

SESSOES_VAL = {
    "2022-03-27-10-30",
    "2022-06-25-10-00",
}

def anotacoes_conflituosas(raiz: Path) -> set[str]:
    """Imagens anotadas mais do que uma vez, com resultados diferentes."""
    por_nome: dict[str, list] = defaultdict(list)
    for split in ("train", "val", "test"):
        ficheiro = raiz / split / PASTA_ANOT / f"pollenbee_{split}.json"
        dados = json.loads(ficheiro.read_text(encoding="utf-8"))
        nome_de = {im["id"]: im["file_name"] for im in dados["images"]}
        caixas: dict[str, list] = defaultdict(list)
        for a in dados["annotations"]:
            caixas[nome_de[a["image_id"]]].append((tuple(a["bbox"]), a["category_id"]))
        for im in dados["images"]:
            por_nome[im["file_name"]].append(tuple(sorted(caixas[im["file_name"]])))
    return {n for n, versoes in por_nome.items() if len(set(versoes)) > 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifesto", type=Path,
                    default=Path("splits/manifest_vnpollenbee.csv"))
    ap.add_argument("--saida", type=Path, default=Path("splits/v1"))
    ap.add_argument("--dataset", type=Path,
                    default=Path("../Datasets/PollenbeeDataset"))
    args = ap.parse_args()

    with args.manifesto.open(encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    # 1. desduplicar: uma linha por imagem única, de forma determinista
    por_md5: dict[str, dict] = {}
    for l in sorted(linhas, key=lambda r: r["ficheiro"]):
        por_md5.setdefault(l["md5"], l)
    unicas = list(por_md5.values())

    # 2. excluir as contraditórias, ANTES de distribuir
    args.saida.mkdir(parents=True, exist_ok=True)
    conflitos = anotacoes_conflituosas(args.dataset)
    antes = len(unicas)
    unicas = [l for l in unicas if Path(l["ficheiro"]).name not in conflitos]
    (args.saida / "excluidas.txt").write_text(
        "# imagens anotadas duas vezes com resultados diferentes\n"
        + "\n".join(sorted(conflitos)) + "\n",
        encoding="utf-8",
    )
    print(f"entradas: {len(linhas)}   únicas: {antes}   "
          f"excluídas: {antes - len(unicas)}   ficam: {len(unicas)}")

    # 3. distribuir
    conjuntos: dict[str, list[dict]] = defaultdict(list)
    for l in unicas:
        conjuntos["val" if l["sessao"] in SESSOES_VAL else "train"].append(l)

    # 4. escrever as listas
    for nome, itens in conjuntos.items():
        (args.saida / f"{nome}.txt").write_text(
            "\n".join(sorted(i["ficheiro"] for i in itens)) + "\n",
            encoding="utf-8",
        )

    # 5. resumo — lendo de volta do disco, não da memória
    for nome in ("train", "val"):
        gravadas = (args.saida / f"{nome}.txt").read_text(encoding="utf-8").split()
        itens = conjuntos[nome]
        caixas = sum(int(i["n_caixas"]) for i in itens)
        print(f"\n{nome}: {len(gravadas)} linhas no ficheiro, "
              f"{caixas} caixas, {caixas / len(itens):.1f}/imagem")
        for s in sorted({i["sessao"] for i in itens}):
            sub = [i for i in itens if i["sessao"] == s]
            c = sum(int(i["n_caixas"]) for i in sub)
            print(f"   {s:<20}{len(sub):>6} img{c:>8} caixas{c / len(sub):>8.1f}/img")

    # 6. verificação
    tr = {i["md5"] for i in conjuntos["train"]}
    va = {i["md5"] for i in conjuntos["val"]}
    if tr & va:
        raise SystemExit(f"ERRO: {len(tr & va)} imagens em treino e validação")
    print("\nsobreposição treino/validação: 0")

if __name__ == "__main__":
    main()