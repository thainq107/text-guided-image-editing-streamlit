import os
import torch
import numpy as np
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchcfm.models.unet import UNetModel
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from sentence_transformers import SentenceTransformer

text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class UNetModelWithTextEmbedding(UNetModel):
    def __init__(self, dim, num_channels, num_res_blocks, embedding_dim, *args, **kwargs):
        super().__init__(dim, num_channels, num_res_blocks, *args, **kwargs)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.embedding_layer = nn.Linear(embedding_dim, num_channels*4)
        self.fc = nn.Linear(num_channels*12, num_channels*4)

    def forward(self, t, x, text_embeddings=None, original_image=None):
        """Apply the model to an input batch, incorporating text embeddings."""
        timesteps = t

        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if (text_embeddings is not None) and (original_image is not None):
            text_embedded = self.embedding_layer(text_embeddings)
            image_embedded = self.image_encoder(original_image).squeeze(2, 3)
            emb = torch.cat([emb, text_embedded, image_embedded], dim=1)
            emb = self.fc(emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
      
model = UNetModelWithTextEmbedding(
    dim=(3, 256, 256), num_channels=32, num_res_blocks=1, embedding_dim=768
)
print(torch.load("text_guided_cfm.pth", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("text_guided_cfm.pth", map_location=torch.device('cpu'), weights_only=False))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.PILToTensor()
])

def euler_method(model, text_embedding, t_steps, dt, noise, original_image):
    y = noise
    y_values = [y]
    with torch.no_grad():
        for t in t_steps[1:]:
            t = t.reshape(-1, )
            dy = model(t, y, text_embeddings=text_embedding, original_image=original_image)
            y = y + dy * dt
            y_values.append(y)
    return torch.stack(y_values)

def inference(original_image, text, model):
    original_image = original_image.unsqueeze(0)
    noise = torch.randn_like(original_image)
    text_embedding = text_encoder.encode(text, convert_to_tensor=True).unsqueeze(0)
    
    # Time parameters
    t_steps = torch.linspace(0, 1, 100)  # Two time steps from 0 to 1
    dt = t_steps[1] - t_steps[0]  # Time step
    
    # Solve the ODE using Euler method
    results = euler_method(model, text_embedding, t_steps, dt, noise, original_image)
    return results[-1]

def main():
    st.title('Text-Guided Image Generation using Conditional Flow Matching')
    st.subheader('Model: Conditional Flow Matching. Dataset: Tedbench')
    text_input = st.text_input("Instruction: ", "A photo of a chair sawed in half.")
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file).convert("RGB").resize((256, 256))
            image = transform(image)
            pred_image, cond_image = inference(image, mask, model)
            grid = make_grid(
                show_imgs.view([-1, 3, 256, 256]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
            )
            img = ToPILImage()(grid)
            st.image(img)
          
    elif option == "Run Example Image":
        image = Image.open('example.png').convert("RGB")
        image = transform(image)
        pred_image = inference(image, text_input, model)
        npimg = pred_image.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg = ((npimg + 1) / 2 * 255).astype(np.uint8)
        st.image(image)
        st.image(npimg, caption="Generated Image", use_column_width=True)

if __name__ == '__main__':
    main() 
