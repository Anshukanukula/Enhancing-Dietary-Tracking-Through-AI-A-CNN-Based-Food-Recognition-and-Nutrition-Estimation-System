# app.py - Streamlit Image Recognition Chatbot with Standard Models
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# =================== Model Loading ===================
@st.cache_resource
def load_model_and_processor():
    """Load the BLIP model and processor with caching"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Loading model on {device}... This might take a few minutes on first run.")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device)
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, device

# =================== Chat Processing ===================

def process_image(image, query="", is_description=True):
    """Process an image with the BLIP model"""
    model, processor, device = load_model_and_processor()
    
    if model is None or processor is None:
        st.error("Model or processor not initialized correctly.")
        return None
    
    try:
        # Resize image to reduce memory usage
        max_size = 1000  # Maximum dimension
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
        
        # Process the image based on mode
        if is_description:
            # For basic description, don't include query
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
                generated_ids = model.generate(
                    **inputs,
                    max_length=75,
                    num_beams=5,
                    min_length=10
                )
                generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Make description more detailed by adding a follow-up analysis
                detailed_query = "What are the main elements, colors, and mood of this image?"
                detailed_inputs = processor(images=image, text=detailed_query, return_tensors="pt").to(device)
                
                follow_up_ids = model.generate(
                    **detailed_inputs,
                    max_length=100,
                    num_beams=5,
                    min_length=20
                )
                follow_up_text = processor.decode(follow_up_ids[0], skip_special_tokens=True)
                
                # Combine both descriptions
                generated_text = f"This image shows {generated_text}. {follow_up_text}"
                
        else:
            # For Q&A mode
            inputs = processor(images=image, text=query, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
                generated_ids = model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=5,
                    min_length=20
                )
                generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # If the answer is too short, try to get more details
                if len(generated_text.split()) < 10 and not generated_text.startswith(query):
                    expand_query = f"Please describe in detail: {query}"
                    expand_inputs = processor(images=image, text=expand_query, return_tensors="pt").to(device)
                    
                    expanded_ids = model.generate(
                        **expand_inputs,
                        max_length=150,
                        num_beams=5,
                        min_length=30
                    )
                    expanded_text = processor.decode(expanded_ids[0], skip_special_tokens=True)
                    
                    # Only use expanded text if it's better than the original
                    if len(expanded_text.split()) > len(generated_text.split()) and not expanded_text.startswith(query):
                        generated_text = expanded_text
                
                # Handle cases where model returns the query verbatim
                if generated_text.startswith(query):
                    fallback_query = "What can you tell me about this?"
                    fallback_inputs = processor(images=image, text=fallback_query, return_tensors="pt").to(device)
                    
                    fallback_ids = model.generate(
                        **fallback_inputs,
                        max_length=100,
                        num_beams=5,
                        min_length=20
                    )
                    fallback_text = processor.decode(fallback_ids[0], skip_special_tokens=True)
                    
                    # Add context from the original query
                    generated_text = f"Regarding {query}: {fallback_text}"
        
        return generated_text
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return f"An error occurred during processing: {e}"

# =================== Main Streamlit App ===================

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Image Chat Assistant",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Add custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .user-message {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .ai-message {
        background-color: #f0f4f8;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stButton > button {
        background-color: #4285f4;
        color: white;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and introduction
    st.title("Image Chat Assistant")
    st.markdown("Upload an image and chat with AI about it. The assistant will automatically provide a description and then you can ask specific questions about the image.")
    
    # Initialize session state for conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'image_uploaded' not in st.session_state:
        st.session_state.image_uploaded = False
        
    if 'automatic_description_generated' not in st.session_state:
        st.session_state.automatic_description_generated = False
    
    # Image upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image to begin the conversation", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the uploaded image
    if uploaded_file is not None:
        try:
            # Store the image in session state
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            
            # If this is the first time we're seeing this image, mark it as uploaded
            if not st.session_state.image_uploaded:
                st.session_state.image_uploaded = True
                st.session_state.automatic_description_generated = False
                # Reset the chat history when a new image is uploaded
                st.session_state.messages = []
                
            # Display the uploaded image in a smaller size
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Current Image", width=300)
                
            # Automatically generate a description for the image when it's first uploaded
            if not st.session_state.automatic_description_generated:
                with st.spinner("Analyzing your image..."):
                    response = process_image(st.session_state.current_image, is_description=True)
                    
                    if response:
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.automatic_description_generated = True
        except Exception as e:
            st.error(f"Error processing your image: {e}")
    
    # Chat interface
    if st.session_state.image_uploaded:
        # Display chat messages
        st.subheader("Chat with your image")
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-message">AI: {message["content"]}</div>', unsafe_allow_html=True)
        
        # Input for new messages
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input("Ask anything about the image:", placeholder="What can you tell me about this image?")
            col1, col2 = st.columns([5, 1])
            with col2:
                submit_button = st.form_submit_button("Send")
            
            if submit_button and query:
                # Add user message to chat display
                st.session_state.messages.append({"role": "user", "content": query})
                
                # Process the query with the model
                with st.spinner("Thinking..."):
                    response = process_image(st.session_state.current_image, query, is_description=False)
                
                # Add AI response to chat display
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                # Rerun to update the UI
                st.rerun()
        
        # Suggestion buttons for common questions
        st.markdown("### Suggested questions:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("What are the main objects in this image?"):
                st.session_state.messages.append({"role": "user", "content": "What are the main objects in this image?"})
                with st.spinner("Thinking..."):
                    response = process_image(st.session_state.current_image, "What are the main objects in this image?", is_description=False)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            if st.button("What is the mood or atmosphere of this scene?"):
                st.session_state.messages.append({"role": "user", "content": "What is the mood or atmosphere of this scene?"})
                with st.spinner("Thinking..."):
                    response = process_image(st.session_state.current_image, "What is the mood or atmosphere of this scene?", is_description=False)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
        with col2:
            if st.button("What colors are prominent in this image?"):
                st.session_state.messages.append({"role": "user", "content": "What colors are prominent in this image?"})
                with st.spinner("Thinking..."):
                    response = process_image(st.session_state.current_image, "What colors are prominent in this image?", is_description=False)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            if st.button("What is unique or interesting about this image?"):
                st.session_state.messages.append({"role": "user", "content": "What is unique or interesting about this image?"})
                with st.spinner("Thinking..."):
                    response = process_image(st.session_state.current_image, "What is unique or interesting about this image?", is_description=False)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    else:
        # Show instructions if no image is uploaded yet
        st.info("👆 Upload an image to start the conversation. The AI will automatically provide a description and then you can ask specific questions about the image.")
    
    # Clear chat button
    if st.session_state.image_uploaded and len(st.session_state.messages) > 0:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.automatic_description_generated = False
            st.rerun()
    
    # Display system information in an expander
    with st.expander("System Information", expanded=False):
        st.write(f"Using device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            st.write(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        st.subheader("Requirements")
        requirements = """
        # requirements.txt
        streamlit==1.31.0
        torch==2.0.1
        transformers==4.36.2
        Pillow==10.0.0
        numpy==1.24.3
        accelerate==0.21.0
        """
        st.code(requirements, language="text")
        
        st.subheader("Installation Instructions")
        install_instructions = """
        # Install dependencies
        pip install -r requirements.txt
        
        # Run the app
        streamlit run app.py
        """
        st.code(install_instructions, language="bash")

if __name__ == "__main__":
    main()