# VIDEO CAPTIONING WEB APP (Streamlit) - updated to remove OpenCV

import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import tempfile
import os
import imageio

# Load model (assume model and vocab already loaded)
class CaptionModel(torch.nn.Module):
    def __init__(self, feat_dim, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim + feat_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, feats, caps):
        embeds = self.embed(caps)
        feats = feats.unsqueeze(1).expand(-1, embeds.size(1), -1)
        x = torch.cat((feats, embeds), dim=2)
        out, _ = self.lstm(x)
        return self.fc(out)

@st.cache_resource
def load_model():
    import json
    with open("vocab.json") as f:
        vocab = json.load(f)
    model = CaptionModel(2048, 256, 512, len(vocab))
    model.load_state_dict(torch.load("trained_caption_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, vocab

@st.cache_resource
def load_resnet():
    resnet = models.resnet50(pretrained=True)
    return torch.nn.Sequential(*list(resnet.children())[:-1])

def generate_caption(model, feature, vocab, max_len=20):
    idx2word = {i: w for w, i in vocab.items()}
    caption = ['<start>']
    input_seq = torch.tensor([[vocab['<start>']]])
    hidden = None
    for _ in range(max_len):
        embed = model.embed(input_seq)
        feat = feature.unsqueeze(0).unsqueeze(1)
        lstm_input = torch.cat((feat, embed), dim=2)
        out, hidden = model.lstm(lstm_input, hidden)
        logits = model.fc(out.squeeze(1))
        predicted = logits.argmax(1).item()
        word = idx2word.get(predicted, '<unk>')
        if word == '<end>':
            break
        caption.append(word)
        input_seq = torch.tensor([[predicted]])
    return ' '.join(caption[1:])

# Streamlit App UI
st.title("🎥 Video Captioning App (Streamlit Cloud Compatible)")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.close()

    try:
        # Read video and extract first frame using imageio
        reader = imageio.get_reader(tfile.name, format='ffmpeg')
        frame = reader.get_data(0)
        image = Image.fromarray(frame)

        # Display image
        st.image(image, caption="Extracted Frame", use_column_width=True)

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        resnet = load_resnet()
        model, vocab = load_model()
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = resnet(tensor).squeeze()
            caption = generate_caption(model, feature, vocab)

        st.success(f"📝 Generated Caption: {caption}")

    except Exception as e:
        st.error(f"❌ Failed to process video. Error: {e}")
