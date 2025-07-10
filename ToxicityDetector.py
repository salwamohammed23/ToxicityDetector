import streamlit as st
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)
from peft import PeftModel, PeftConfig
from PIL import Image
import torch
import os

# App setup
st.set_page_config(
    page_title="Advanced Content Safety Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🛡️ Advanced Content Safety Analyzer")

# Load models
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    try:
        # Image captioning model
        with st.spinner("Loading image captioning model..."):
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Initial check model
        with st.spinner("Loading initial check model..."):
            flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # Detailed analysis model
        with st.spinner("Loading detailed analysis model..."):
            model_path = "Model/lora_distilbert_toxic_final"
            config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=9,
                return_dict=True,
                ignore_mismatched_sizes=True
            )
            lora_model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_model.to(device)
        lora_model.to(device)
        
        return blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device
    
    except Exception as e:
        st.error(f"An error occurred while loading the models: {str(e)}")
        return None, None, None, None, None, None

# Label definitions
LABELS = {
    0: {"name": "Safe", "emoji": "✅", "color": "green"},
    1: {"name": "Hate Speech", "emoji": "💢", "color": "red"},
    2: {"name": "Insult", "emoji": "🗯️", "color": "orange"},
    3: {"name": "Threat", "emoji": "⚠️", "color": "red"},
    4: {"name": "Racist", "emoji": "🚫", "color": "red"},
    5: {"name": "Sexual", "emoji": "🔞", "color": "red"},
    6: {"name": "Incitement", "emoji": "🔥", "color": "orange"},
    7: {"name": "Other", "emoji": "❓", "color": "gray"},
    8: {"name": "Self-harm", "emoji": "💔", "color": "red"}
}

# Initial safety check function
def initial_safety_check(text, flan_pipe):
    prompt = f"Is this content safe or unsafe? \"{text}\" Answer with one word only: Safe or Unsafe."
    result = flan_pipe(prompt, max_new_tokens=10)
    return result[0]['generated_text'].strip()

# Detailed analysis function
def detailed_analysis(text, lora_model, tokenizer, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = lora_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return probs[0].tolist()

def main():
    blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device = load_models()
    
    # Refresh button
    if st.button("🔄 Refresh Page"):
        st.experimental_rerun()
    
    input_type = st.radio(
        "Select content type:",
        ["Text", "Image"],
        horizontal=True,
        key="input_type"
    )
    
    if input_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload an image for analysis:",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image", key="analyze_image"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Generate caption
                        raw_image = Image.open(uploaded_file).convert("RGB")
                        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
                        out = blip_model.generate(**inputs)
                        caption = blip_processor.decode(out[0], skip_special_tokens=True)
                        
                        st.success(f"**Caption:** {caption}")
                        
                        # Initial check
                        st.subheader("🔍 Initial Safety Check")
                        initial_check = initial_safety_check(caption, flan_pipe)
                        
                        if "unsafe" in initial_check.lower():
                            st.error("## ❌ Initial Check Result: Unsafe Content")
                            st.error("Unsafe content detected in the initial check. Analysis stopped.")
                            st.stop()
                        else:
                            st.success("## ✅ Initial Check Result: Safe Content")
                            
                            # Detailed analysis
                            st.subheader("🔎 Detailed Analysis")
                            probs = detailed_analysis(caption, lora_model, tokenizer, device)
                            pred_idx = probs.index(max(probs))
                            confidence = probs[pred_idx]
                            label = LABELS[pred_idx]
                            
                            st.markdown(f"""
                            <div style='background-color:#f0f0f0; padding:15px; border-radius:10px; border-left:5px solid {label["color"]}'>
                                <h3 style='color:{label["color"]}'>{label["emoji"]} Category: <strong>{label["name"]}</strong></h3>
                                <p>Confidence Level: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("### Probability Distribution:")
                            for i, prob in enumerate(probs):
                                label_info = LABELS[i]
                                cols = st.columns([1, 3, 1])
                                cols[0].markdown(f"**{label_info['emoji']} {label_info['name']}**")
                                cols[1].progress(prob, text=f"{prob:.2%}")
                                cols[2].write(f"{prob:.2%}")
                            
                    except Exception as e:
                        st.error(f"An error occurred while analyzing the image: {str(e)}")
    
    elif input_type == "Text":
        text_content = st.text_area(
            "Enter text for analysis:",
            height=200,
            placeholder="Paste your text here...",
            key="text_input"
        )
        
        if st.button("Analyze Text", key="analyze_text"):
            if not text_content.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text..."):
                    try:
                        # Initial check
                        st.subheader("🔍 Initial Safety Check")
                        initial_check = initial_safety_check(text_content, flan_pipe)
                        
                        if "unsafe" in initial_check.lower():
                            st.error("## ❌ Initial Check Result: Unsafe Content")
                            st.error("Unsafe content detected in the initial check. Analysis stopped.")
                            st.stop()
                        else:
                            st.success("## ✅ Initial Check Result: Safe Content")
                            
                            # Detailed analysis
                            st.subheader("🔎 Detailed Analysis")
                            probs = detailed_analysis(text_content, lora_model, tokenizer, device)
                            pred_idx = probs.index(max(probs))
                            confidence = probs[pred_idx]
                            label = LABELS[pred_idx]
                            
                            st.markdown(f"""
                            <div style='background-color:#f0f0f0; padding:15px; border-radius:10px; border-left:5px solid {label["color"]}'>
                                <h3 style='color:{label["color"]}'>{label["emoji"]} Category: <strong>{label["name"]}</strong></h3>
                                <p>Confidence Level: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("### Probability Distribution:")
                            for i, prob in enumerate(probs):
                                label_info = LABELS[i]
                                cols = st.columns([1, 3, 1])
                                cols[0].markdown(f"**{label_info['emoji']} {label_info['name']}**")
                                cols[1].progress(prob, text=f"{prob:.2%}")
                                cols[2].write(f"{prob:.2%}")
                    
                    except Exception as e:
                        st.error(f"An error occurred while analyzing the text: {str(e)}")

if __name__ == "__main__":
    main()
