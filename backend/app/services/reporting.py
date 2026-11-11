"""Relatórios portáteis: tabelas e curvas SVG, sem dependências externas."""

from html import escape


def display(value, percent=False):
    if value is None:
        return "Não disponível"
    if percent:
        return f"{value * 100:.2f}%".replace(".", ",")
    return f"{value:.6f}".replace(".", ",") if isinstance(value, float) else str(value)


def curve(run, metric):
    title = "Acurácia × Épocas" if metric == "accuracy" else "Loss × Épocas"
    history = run["history"]
    if not history:
        return "<p>Histórico ainda não disponível.</p>"
    values = [epoch[key] for epoch in history for key in (metric, f"val_{metric}")]
    upper = 1 if metric == "accuracy" else max(max(values) * 1.1, 0.001)
    first, last = history[0]["epoch"], history[-1]["epoch"]

    def x(epoch):
        return 72 + (epoch - first) / max(1, last - first) * 600

    def y(value):
        return 280 - value / upper * 230

    content = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 360" role="img" '
        f'aria-label="{title}"><title>Curva de aprendizado — {title}</title>'
    )
    for index in range(6):
        value = upper * index / 5
        label = f"{value * 100:.0f}%" if metric == "accuracy" else f"{value:.3f}"
        content += (
            f'<path d="M72 {y(value):.2f} H672" stroke="#e2e8f0"/>'
            f'<text x="65" y="{y(value) + 4:.2f}" text-anchor="end">{label}</text>'
        )
    for epoch in sorted({first, last, (first + last) // 2}):
        content += f'<text x="{x(epoch):.2f}" y="302" text-anchor="middle">{epoch}</text>'
    saved = run.get("saved_epoch")
    if saved is not None:
        content += (
            f'<path d="M{x(saved):.2f} 40 V280" stroke="#64748b" stroke-dasharray="3 4"/>'
            f'<text x="{x(saved):.2f}" y="28" text-anchor="middle">'
            f'Pesos salvos: época {saved}</text>'
        )
    for key, color, dash, label, legend in [
        (metric, "#315ddb", "", "Treino", 180),
        (f"val_{metric}", "#107b73", "7 4", "Validação", 390),
    ]:
        points = " ".join(f'{x(e["epoch"]):.2f},{y(e[key]):.2f}' for e in history)
        content += (
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-dasharray="{dash}" points="{points}"/>'
        )
        content += "".join(
            f'<circle cx="{x(e["epoch"]):.2f}" cy="{y(e[key]):.2f}" r="2" fill="{color}"/>'
            for e in history
        )
        content += (
            f'<path d="M{legend} 345 h30" stroke="{color}" stroke-width="2" '
            f'stroke-dasharray="{dash}"/><text x="{legend + 38}" y="349">{label}</text>'
        )
    ylabel = "Acurácia (%)" if metric == "accuracy" else "Loss"
    return content + (
        '<text x="372" y="324" text-anchor="middle">Épocas</text>'
        '<text transform="translate(17 170) rotate(-90)" text-anchor="middle">'
        f'{ylabel}</text></svg>'
    )


def render_report(run, format_name):
    html = format_name == "html"
    sections = []

    def heading(title, level=2):
        sections.append(
            f"<h{level}>{escape(title)}</h{level}>" if html else f"{'#' * level} {title}\n"
        )

    def paragraph(text):
        sections.append(f"<p>{escape(text)}</p>" if html else text + "\n")

    def table(headers, rows):
        if html:
            sections.append("<div class='table-scroll'><table><thead><tr>" + "".join(
                f"<th scope='col'>{escape(str(cell))}</th>" for cell in headers
            ) + "</tr></thead><tbody>" + "".join(
                "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
                for row in rows
            ) + "</tbody></table></div>")
        else:
            def line(row):
                cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
                return "| " + " | ".join(cells) + " |"
            lines = [line(headers), line(["---"] * len(headers)), *map(line, rows)]
            sections.append("\n".join(lines) + "\n")

    test = run.get("test_metrics")
    metrics = test or run.get("validation_metrics")
    subset = "teste final" if test else "validação"
    heading(f"Relatório de classificação — {subset}")
    if metrics:
        report = metrics.get("classification_report")
        if report:
            def class_row(key, label):
                return [label, *[display(report[key][metric], True)
                                 for metric in ("precision", "recall", "f1-score")],
                        int(report[key]["support"])]

            table(["Categoria / Métrica", "Precision", "Recall", "F1-score", "Support"], [
                class_row("0", "Classe 0"),
                class_row("1", "Classe 1"),
                ["Acurácia", "—", "—", display(metrics["accuracy"], True), metrics["samples"]],
                class_row("macro avg", "Média macro"),
                class_row("weighted avg", "Média ponderada"),
            ])
        else:
            paragraph("Relatório por classe não disponível neste experimento antigo.")
    else:
        paragraph("Métricas ainda não disponíveis. Aguarde a conclusão do treinamento.")
    heading(f"Matriz de confusão — {subset}")
    if metrics:
        table(["Classe real ↓ / Prevista →", "Classe 0", "Classe 1"], [
            ["Classe 0", metrics["tn"], metrics["fp"]],
            ["Classe 1", metrics["fn"], metrics["tp"]],
        ])
    else:
        paragraph("Matriz ainda não disponível. Aguarde a conclusão do treinamento.")
    for metric, title in [
        ("accuracy", "Curva de aprendizado — Acurácia × Épocas"),
        ("loss", "Curva de aprendizado — Loss × Épocas"),
    ]:
        heading(title)
        sections.append(curve(run, metric))
    body = "\n\n".join(sections)
    if not html:
        return body
    return '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório de classificação</title><style>
body{font:15px/1.6 system-ui,sans-serif;color:#18263d;background:#f3f6fb;margin:0}
main{max-width:1000px;margin:32px auto;padding:36px;background:white;border-radius:16px}
h2{line-height:1.25;margin-top:36px;color:#000}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:12px;text-align:left;border-bottom:1px solid #dce3ed;overflow-wrap:anywhere;
background:#fff}th{color:#000}
th:first-child,td:first-child{min-width:150px}
.table-scroll{overflow-x:auto}
svg{width:100%;height:auto;font:12px system-ui;break-inside:avoid}p{overflow-wrap:anywhere}
@media print{body{background:white}main{margin:0;padding:0;max-width:none}
h2{break-after:avoid}tr{break-inside:avoid}}
@media(max-width:600px){main{padding:16px;margin:0}}
</style><main>''' + body + "</main></html>"
