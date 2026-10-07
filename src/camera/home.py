import streamlit as st

st.title("Project name")

c1, c2 = st.columns(2, gap="small", vertical_alignment="center")

with c1:
    st.image("Image", width=300)
with c2:
    st.title("", anchor=False)

    st.write("""
        """
    )
