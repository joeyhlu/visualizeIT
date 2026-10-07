import streamlit as st

def main():
    # Page set up
    home_page = st.Page(
        page="src/camera/home.py", 
        title="Home", 
        icon="🏠",
        default=True,
    )

    video_page = st.Page(
        page="src/camera/video.py", 
        title="Application", 
        icon="📷"
    )


    page = st.navigation(pages=[home_page, video_page])
    st.sidebar.text("")

    page.run()



main()


 
