import os
import pickle

import streamlit as st
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model(model_dir_path):
    with open(f'{model_dir_path}/vectorizer.pkl', 'rb') as pickle_file:
        vectorizer = pickle.load(pickle_file)
    with open(f'{model_dir_path}/classifier.pkl', 'rb') as pickle_file:
        classifier = pickle.load(pickle_file)

    return vectorizer, classifier


if __name__ == '__main__':
    st.set_page_config('Poem Analyzer')

    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    model_dir_path = 'linear/data'
    vectorizer, classifier = load_model(model_dir_path)

    st.markdown("<h1 style='text-align: center;'>Poem Analyzer</h1>", unsafe_allow_html=True)

    img_logo_path = os.path.join(os.path.dirname(__file__), 'icon.png')
    logo_image = Image.open(img_logo_path)
    st.image(logo_image, width=200)

    poem2index = {'حافظ': 0, 'مولوی': 1, 'سعدی': 2, 'عطار': 3, 'سنایی': 4, 'صائب تبریزی': 5, 'رشیدالدین میبدی': 6}
    index2poem = {index: poem for poem, index in poem2index.items()}

    poem = st.text_area(label='Poem:')

    if st.button(label='Run'):

        if poem == '':
            st.markdown("<p style='color: red;'>Fill poem field please!</p>", unsafe_allow_html=True)
        else:
            feature_vector = vectorizer.transform([poem])
            result = classifier.predict_proba(feature_vector)
            for index in index2poem.keys():
                st.markdown(f'<p>{index2poem[index]}: {round(result[0][index], 2)}</p>', unsafe_allow_html=True)
