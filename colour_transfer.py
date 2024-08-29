import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import ot
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from io import BytesIO
from PIL import Image

rng = np.random.RandomState(42)


@st.cache_data
def load_image_from_url(url, max_retries=3):
    session = requests.Session()
    retry = Retry(
        total=max_retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img).astype(np.float64) / 255.0
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


def colour_transport(I1, I2, nb=500):
    X1 = im2mat(I1)
    X2 = im2mat(I2)

    # take sample
    idx1 = rng.randint(X1.shape[0], size=(nb,))
    idx2 = rng.randint(X2.shape[0], size=(nb,))
    Xs = X1[idx1, :]
    Xt = X2[idx2, :]

    # transport computation
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    transp_Xs_emd = ot_emd.transform(Xs=X1)
    I1t = minmax(mat2im(transp_Xs_emd, I1.shape))

    return I1t


# Initialize session state variables if they don't exist
if "source_url" not in st.session_state:
    st.session_state.source_url = ""
if "target_url" not in st.session_state:
    st.session_state.target_url = ""


# Function to swap URLs
def swap_urls():
    st.session_state.source_url, st.session_state.target_url = (
        st.session_state.target_url,
        st.session_state.source_url,
    )


# Input fields
col1, col2 = st.columns([3, 1])
with col1:
    source_url = st.text_input("Image to modify:", key="source_url")
    target_url = st.text_input("Source style :", key="target_url")
with col2:
    st.button("Swap URLs", on_click=swap_urls)

source_img = None
target_img = None
result_img = None


if source_url and target_url:
    source_img = load_image_from_url(source_url)
    target_img = load_image_from_url(target_url)

    if source_img is not None and target_img is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(source_img, caption="Target", use_column_width=True)

        with col3:
            st.image(target_img, caption="Source", use_column_width=True)

        result_img = colour_transport(source_img, target_img)

        with col2:
            st.image(result_img, caption="Style Transfer Result", use_column_width=True)
    else:
        st.error(
            "Failed to load one or both images. Please check the URLs and try again."
        )

st.markdown(
    "Note: Please ensure that the image URLs are direct links to the image files. If you encounter issues, try a different image host."
)
