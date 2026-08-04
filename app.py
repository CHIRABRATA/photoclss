"""Main Streamlit application entry point."""
import streamlit as st

def main():
    st.set_page_config(page_title="Trip Photo Organizer AI", layout="wide")
    st.title("📸 Trip Photo Organizer AI")
    st.write("Environment and Streamlit setup successfully completed!")

if __name__ == "__main__":
    main()