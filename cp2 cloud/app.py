import base64
from io import BytesIO
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import requests
import streamlit as st

st.set_page_config(
    page_title="Metria Vision - Reconhecimento de Imagens",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;a
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #f0f2f6;
    }
    .sub-caption {
        font-size: 0.95rem;
        color: #a0a5b5;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 6px;
        background-color: #1e2638;
        color: #dce3f1;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #323d57;
    }
    .theme-box {
        padding: 12px 16px;
        background-color: #161d2d;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_KEY = os.getenv(
    "GOOGLE_API_KEY", "AIzaSyCRmzzvCq4xKJ8mCoGiKUZUKM8JPiiiRog"
).strip()


def get_first(sequence, default=None):
    """Retorna com seguranca o primeiro item sem usar colchetes numericos."""
    if not sequence:
        return default
    for item in sequence:
        return item
    return default


def translate_to_pt(text: str) -> str:
    """Traduz termos em ingles para portugues via requisicao direta."""
    if not text:
        return ""
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "pt",
            "dt": "t",
            "q": text,
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            nivel_1 = get_first(data, ())
            nivel_2 = get_first(nivel_1, ())
            traducao = get_first(nivel_2, text)
            return str(traducao)
    except Exception:
        pass
    return text


def identify_theme(labels: List[str], text_content: str) -> str:
    """Classifica o tema central da imagem com base nos termos extraidos."""
    corpus = (" ".join(labels) + " " + text_content).lower()

    if any(k in corpus for k in ["circuit", "logo", "clp", "plaque", "electronic", "wire", "schematic", "electric"]):
        return "Engenharia, Eletroeletronica e Automacao"
    if any(k in corpus for k in ["fluxo", "compras", "process", "flowchart", "gestao", "hospital", "saude", "relatorio", "invoice"]):
        return "Processos Organizacionais e Gestao Administrativa"
    if any(k in corpus for k in ["diagram", "plan", "architecture", "software", "system", "code"]):
        return "Tecnologia da Informacao e Arquitetura de Sistemas"
    if any(k in corpus for k in ["vehicle", "car", "transport", "road"]):
        return "Mobilidade e Transportes"
    if any(k in corpus for k in ["people", "person", "crowd", "human"]):
        return "Sociedade e Pessoas"
    return "Tecnologia, Dados e Comunicacao Visual"


def generate_description(theme: str, main_label: str, top_tags: List[str], has_text: bool) -> str:
    """Elabora uma sintese textual formal sobre a composicao da imagem."""
    tags_amostra = []
    for tag in top_tags:
        tags_amostra.append(tag)
        if len(tags_amostra) >= 4:
            break

    tags_str = ", ".join(tags_amostra) if tags_amostra else "elementos graficos"
    desc = "A imagem analisada enquadra-se na categoria de " + theme + ". "
    desc += "O elemento predominante identificado e classificado como '" + main_label + "', "
    desc += "apresentando correlacao direta com " + tags_str + ". "
    if has_text:
        desc += "Foi identificado conteudo textual estruturado que pode ser extraido via OCR para analise de fluxo de trabalho ou documentacao."
    else:
        desc += "Nao foram detectados volumes significativos de texto sobrepostos aos elementos graficos."
    return desc


def analyze_image_vision(
    image_bytes: Optional[bytes] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Envia a imagem para a API do Google Cloud Vision."""
    endpoint = "https://vision.googleapis.com/v1/images:annotate?key=" + API_KEY

    if not image_bytes:
        return False, {"error": "Nenhuma imagem fornecida."}

    encoded_img = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [
            {
                "image": {"content": encoded_img},
                "features": [
                    {"type": "LABEL_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 10},
                ],
            }
        ]
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        return False, {
            "status_code": response.status_code,
            "detail": response.text,
        }
    except Exception as exc:
        return False, {"error": str(exc)}


def main() -> None:
    st.markdown('<div class="main-title">Sistema de Reconhecimento e Analise Visual</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-caption">Google Cloud Platform - Vision API | Servicos Cognitivos em Nuvem</div>', unsafe_allow_html=True)

    input_type = st.radio(
        "Origem da imagem:",
        options=["Upload de Arquivo Local", "URL da Imagem na Nuvem"],
        horizontal=True,
    )

    image_bytes: Optional[bytes] = None
    display_image: Optional[Image.Image] = None

    if input_type == "Upload de Arquivo Local":
        uploaded_file = st.file_uploader(
            "Selecione uma imagem (formatos compativeis: JPG, PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            display_image = Image.open(BytesIO(image_bytes))
    else:
        url_input = st.text_input(
            "Endereco URL da imagem:",
            placeholder="https://exemplo.com/imagem.jpg",
        )
        if url_input.strip():
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url_input.strip(), headers=headers, timeout=10)
                if resp.status_code == 200:
                    image_bytes = resp.content
                    display_image = Image.open(BytesIO(image_bytes))
            except Exception:
                st.warning("Nao foi possivel carregar o arquivo a partir da URL informada.")

    if display_image:
        st.image(
            display_image, caption="Previa do Arquivo Carregado", use_container_width=True
        )

        if st.button("Executar Analise Cognitiva", type="primary"):
            with st.spinner("Realizando processamento na nuvem..."):
                success, result = analyze_image_vision(image_bytes=image_bytes)

            if success:
                st.success("Analise concluida com exito.")

                responses_list = result.get("responses", ())
                first_resp = get_first(responses_list, {})

                labels = first_resp.get("labelAnnotations", ())
                texts = first_resp.get("textAnnotations", ())

                raw_tags = []
                for item in labels:
                    raw_tags.append(item.get("description", ""))

                first_text_obj = get_first(texts, {})
                full_text = first_text_obj.get("description", "")

                # Traduz as tags para o portugues
                pt_tags = []
                for tag_en in raw_tags:
                    pt_tags.append(translate_to_pt(tag_en))

                # Identifica o tema formal da imagem
                identified_theme = identify_theme(raw_tags, full_text)

                st.divider()

                # Bloco visual do tema central
                st.markdown(
                    '<div class="theme-box"><strong>Tema Identificado:</strong> ' + identified_theme + '</div>',
                    unsafe_allow_html=True,
                )

                # Classificacao principal
                if labels:
                    top_label_obj = get_first(labels, {})
                    top_label_en = top_label_obj.get("description", "Indefinido")
                    top_label_pt = get_first(pt_tags, top_label_en)
                    confidence = float(top_label_obj.get("score", 0.0))

                    col1, col2 = st.columns((2, 1))
                    with col1:
                        st.subheader("Classificacao Principal")
                        st.markdown("### **" + top_label_pt.capitalize() + "**")
                        st.caption("Termo original: " + top_label_en)
                    with col2:
                        st.metric(
                            label="Grau de Confianca",
                            value=str(round(confidence * 100, 1)) + "%",
                        )
                        st.progress(confidence)

                    # Descricao sintetica formal do conteudo
                    st.write("")
                    st.subheader("Descricao Sintetica do Conteudo")
                    description_text = generate_description(
                        identified_theme, top_label_pt, pt_tags, bool(texts)
                    )
                    st.info(description_text)

                    # Badges com rotulos em portugues
                    st.write("")
                    st.subheader("Rotulos e Categorias Identificadas (PT-BR)")
                    badges_html_parts = []
                    for pt_t, lbl_obj in zip(pt_tags, labels):
                        score_num = round(float(lbl_obj.get("score", 0.0)) * 100)
                        item_html = '<span class="badge">' + pt_t.capitalize() + ' (' + str(score_num) + '%)</span>'
                        badges_html_parts.append(item_html)

                    st.markdown("".join(badges_html_parts), unsafe_allow_html=True)
                else:
                    st.info("Nenhum rotulo categorico foi associado a imagem.")

                # Extracao de Texto (OCR)
                if texts:
                    st.write("")
                    st.subheader("Extracao Textual (OCR)")
                    st.text_area(
                        label="Conteudo textual detectado:",
                        value=full_text,
                        height=160,
                        help="Transcricao do texto presente na imagem",
                    )

                with st.expander("Registro Tecnico JSON da Resposta"):
                    st.json(result)
            else:
                st.error("Falha na requisicao: " + str(result))


if __name__ == "__main__":
    main()