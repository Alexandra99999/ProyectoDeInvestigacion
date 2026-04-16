from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np, re, os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Modelo ligero y rápido
# ----------------------------
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Importar microcurrículos
from microcurriculos import microcurriculos

# ----------------------------
# Preprocesamiento de textos
# ----------------------------
stopwords = set([
    "materia","curso","asignatura","tema","unidad","aprendizaje","estudiante","estudio","docente",
    "clase","contenido","objetivo","objetivos","teoría","práctica","introducción"
])

def limpiar_texto(texto):
    texto = texto.lower()
    palabras = re.findall(r'\b[a-záéíóúñ]+\b', texto)
    palabras = [p for p in palabras if p not in stopwords]
    return " ".join(palabras)

# ----------------------------
# Sentiment Analysis (reglas simples)
# ----------------------------
palabras_positivas = {
    "excelente","bueno","positivo","satisfecho","feliz","exitoso","mejorar","agradable",
    "interesante","útil","fácil","bien","logrado"
}

palabras_negativas = {
    "malo","negativo","triste","aburrido","fracaso","difícil","estresado","terrible",
    "pésimo","preocupado","problema","confuso","complicado","frustrado","frustra","urgente"
}

def analizar_sentimiento(texto):
    texto = texto.lower()
    tokens = re.findall(r'\b[a-záéíóúñ]+\b', texto)
    pos = sum(1 for t in tokens if t in palabras_positivas)
    neg = sum(1 for t in tokens if t in palabras_negativas)

    if pos == 0 and neg == 0:
        return "Neutral"
    elif neg >= pos * 2:
        return "Muy Negativo"
    elif neg > pos:
        return "Negativo"
    elif pos >= neg * 2:
        return "Muy Positivo"
    else:
        return "Positivo"

# ----------------------------
# Preparar textos
# ----------------------------
materias = list(microcurriculos.keys())
textos = [
    limpiar_texto(
        microcurriculos[m].get("definicion", "") + " " +
        " ".join(microcurriculos[m].get("resultados_aprendizaje", []))
    )
    for m in materias
]

# ----------------------------
# Embeddings
# ----------------------------
materia_embeddings = model.encode(textos, show_progress_bar=False).astype("float32")

# PCA adaptativo
n_comp = min(30, materia_embeddings.shape[0], materia_embeddings.shape[1])
pca = PCA(n_components=n_comp, random_state=42)
materia_embeddings_pca = pca.fit_transform(materia_embeddings)

# ----------------------------
# Clustering optimizado (KMeans + método del codo)
# ----------------------------
def evaluar_kmeans(emb):
    resultados = []
    best_k, best_ch, best_inertia, best_labels = None, -1, None, None
    emb_scaled = StandardScaler().fit_transform(emb)

    for k in range(2, min(10, emb.shape[0])):
        km = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = km.fit_predict(emb_scaled)
        if len(set(labels)) < 2:
            continue
        ch = calinski_harabasz_score(emb_scaled, labels)
        inertia = km.inertia_
        resultados.append([k, inertia, ch])  # guardamos todo

        if ch > best_ch:  # elegimos el modelo con mayor CH
            best_k, best_ch, best_inertia, best_labels = k, ch, inertia, labels

    return {
        "method": f"KMeans(k={best_k})" if best_k else "N/A",
        "calinski_harabasz": best_ch,
        "inertia": best_inertia,
        "labels": best_labels,
        "codo": resultados  # guardamos todos los valores para el reporte
    }

def evaluar_dbscan(emb):
    emb_scaled = StandardScaler().fit_transform(emb)
    neigh = NearestNeighbors(n_neighbors=2).fit(emb_scaled)
    dists, _ = neigh.kneighbors(emb_scaled)
    dist_sort = np.sort(dists[:,1])
    eps_candidates = [np.percentile(dist_sort,p) for p in [60,70,80,90]]
    best = {"method":"DBSCAN","calinski_harabasz":-1,"assigned_ratio":0,"labels":None}
    for eps in eps_candidates:
        for min_s in [2,3,4]:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(emb_scaled)
            if len(set(labels))<2 or np.sum(labels!=-1)<5: 
                continue
            assigned_ratio = np.mean(labels!=-1)
            try:
                ch = calinski_harabasz_score(emb_scaled[labels!=-1], labels[labels!=-1])
            except:
                ch = -1
            if ch > best["calinski_harabasz"]:
                best = {
                    "method": f"DBSCAN(eps={eps:.3f},min={min_s})",
                    "calinski_harabasz": ch,
                    "assigned_ratio": assigned_ratio,
                    "labels": labels
                }
    return best

def evaluar_clustering(emb):
    res_kmeans = evaluar_kmeans(emb)
    res_dbscan = evaluar_dbscan(emb)
    # elegir el que tenga mejor CH Index
    return res_kmeans if res_kmeans["calinski_harabasz"] >= res_dbscan["calinski_harabasz"] else res_dbscan

# ----------------------------
# Recomendador
# ----------------------------
def recomendar_por_perfil(perfil_texto, top_n=5):
    perfil_clean = limpiar_texto(perfil_texto)
    emb_perfil = model.encode([perfil_clean], show_progress_bar=False).astype("float32")
    similitudes = cosine_similarity(emb_perfil, materia_embeddings)[0]
    orden = np.argsort(-similitudes)
    recomendaciones = []
    for idx in orden[:top_n]:
        rec = {
            "materia": materias[idx],
            "similitud": float(similitudes[idx]),
            "definicion": microcurriculos[materias[idx]].get("definicion", ""),
            "contenido": microcurriculos[materias[idx]].get("contenido", []),
            "herramientas": microcurriculos[materias[idx]].get("herramientas", []),
            "referencias": microcurriculos[materias[idx]].get("referencias", []),
            "resultados_aprendizaje": microcurriculos[materias[idx]].get("resultados_aprendizaje", []),
        }
        recomendaciones.append(rec)
    return recomendaciones

# ----------------------------
# PDF Reporte (con logo en todas las páginas y sentimiento)
# ----------------------------
def generar_reporte_pdf(codigo, nombre, perfil, recomendaciones, clustering, filename):
    def header_footer(canvas, doc):
        logo_path = "Fup.jpeg"
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, doc.pagesize[0]-140, doc.pagesize[1]-80, width=120, height=60)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 20, f"Fundación Universitaria de Popayán — {datetime.now().strftime('%d/%m/%Y')}")

    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    titulo = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.darkblue)

    # Encabezado
    story.append(Paragraph("REPORTE DE RECOMENDACIÓN ACADÉMICA", titulo))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Código: {codigo}  |  Nombre: {nombre}  |  Fundación Universitaria de Popayán", styles['Normal']))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Perfil
    story.append(Paragraph("Perfil ingresado:", styles['Heading2']))
    story.append(Paragraph(perfil, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Análisis de Sentimiento
    sentimiento = analizar_sentimiento(perfil)
    story.append(Paragraph("Análisis de Sentimiento:", styles['Heading2']))
    story.append(Paragraph(f"Sentimiento detectado: <b>{sentimiento}</b>", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Clustering
    story.append(Paragraph("Clustering (mejor modelo):", styles['Heading2']))
    if "inertia" in clustering:  # KMeans
        story.append(Paragraph(f"{clustering['method']}", styles['BodyText']))
        story.append(Paragraph(f"Calinski–Harabasz Index: {clustering['calinski_harabasz']:.3f}", styles['BodyText']))
        story.append(Paragraph(f"Inertia (WCSS): {clustering['inertia']:.3f}", styles['BodyText']))

        # Método del Codo con tabla extendida
        story.append(Spacer(1, 12))
        story.append(Paragraph("Método del Codo (KMeans):", styles['Heading3']))

        table_data = [["k", "Inertia", "Calinski–Harabasz"]]
        for k, inertia, ch in clustering["codo"]:
            table_data.append([str(k), f"{inertia:.4f}", f"{ch:.4f}"])

        table = Table(table_data, colWidths=[0.7*inch, 1.5*inch, 2.0*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(table)
    else:  # DBSCAN
        story.append(Paragraph(f"{clustering['method']}", styles['BodyText']))
        story.append(Paragraph(f"Calinski–Harabasz Index: {clustering['calinski_harabasz']:.3f}", styles['BodyText']))
        story.append(Paragraph(f"Porcentaje de puntos asignados: {clustering['assigned_ratio']*100:.1f}%", styles['BodyText']))

    # Página nueva con recomendaciones
    story.append(PageBreak())
    story.append(Paragraph("Materias recomendadas:", titulo))
    for i, rec in enumerate(recomendaciones, 1):
        story.append(Paragraph(f"{i}. {rec['materia']} — Similitud {rec['similitud']:.3f}", styles['Heading3']))
        story.append(Paragraph("<b>Definición:</b> " + rec['definicion'], styles['BodyText']))

        # Contenido con temas y subtemas
        contenido = rec.get("contenido", [])
        story.append(Paragraph("<b>Contenido:</b>", styles['Normal']))
        if isinstance(contenido, list) and contenido:
            for t_idx, tema in enumerate(contenido, 1):
                if isinstance(tema, dict) and "titulo" in tema:
                    story.append(Paragraph(f"{t_idx}. {tema['titulo']}", styles['BodyText']))
                    for sub in tema.get("subtemas", []):
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;• {sub}", styles['BodyText']))
                elif isinstance(tema, str):
                    story.append(Paragraph(f"{t_idx}. {tema}", styles['BodyText']))

        # Resultados de aprendizaje
        if rec['resultados_aprendizaje']:
            story.append(Paragraph("<b>Resultados de Aprendizaje:</b>", styles['Normal']))
            for j, ra in enumerate(rec['resultados_aprendizaje'], 1):
                story.append(Paragraph(f"{j}. {ra}", styles['BodyText']))

        # Herramientas
        if rec["herramientas"]:
            story.append(Paragraph("<b>Herramientas:</b>", styles['Normal']))
            for h in rec["herramientas"]:
                story.append(Paragraph(f"• {h}", styles['BodyText']))

        # Referencias
        if rec["referencias"]:
            story.append(Paragraph("<b>Referencias:</b>", styles['Normal']))
            for r in rec["referencias"]:
                story.append(Paragraph(f"• {r}", styles['BodyText']))

        story.append(Spacer(1, 12))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

# ----------------------------
# Main
# ----------------------------
def main():
    print("📝 Sistema de recomendación optimizado y ligero")
    codigo = input("Código de estudiante: ")
    nombre = input("Nombre: ")
    perfil = input("Describe tu perfil académico: ")
    recomendaciones = recomendar_por_perfil(perfil, top_n=5)
    clustering = evaluar_clustering(materia_embeddings_pca)
    if not os.path.exists("reportes"): 
        os.makedirs("reportes")
    filename = f"reportes/reporte_{codigo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    generar_reporte_pdf(codigo, nombre, perfil, recomendaciones, clustering, filename)
    print("\n✅ PDF generado en", filename)
    print("Mejor modelo:", clustering["method"])
    if "inertia" in clustering:  # KMeans
        print("Calinski–Harabasz:", clustering["calinski_harabasz"])
        print("Inertia:", clustering["inertia"])
    else:  # DBSCAN
        print("Calinski–Harabasz:", clustering["calinski_harabasz"])
        print("Porcentaje asignado:", clustering["assigned_ratio"])

if __name__ == "__main__":
    main()
