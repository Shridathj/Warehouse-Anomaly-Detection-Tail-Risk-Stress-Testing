import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
          /* Layout */
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
          }

          /* Hero */
          .dashboard-hero {
            background: linear-gradient(135deg, #1a5276 0%, #2471a3 55%, #2e86c1 100%);
            border-radius: 14px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.75rem;
            color: #ffffff;
          }
          .dashboard-hero h1 {
            margin: 0 0 0.5rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #ffffff !important;
          }
          .dashboard-hero p {
            margin: 0;
            opacity: 0.92;
            font-size: 1.02rem;
            line-height: 1.55;
          }

          /* Scenario header */
          .scenario-banner {
            background: #f4f6f9;
            border: 1px solid #dde3ea;
            border-left: 5px solid #1a5276;
            border-radius: 10px;
            padding: 1.1rem 1.35rem;
            margin-bottom: 1.5rem;
          }
          .scenario-banner h2 {
            margin: 0 0 0.35rem 0;
            font-size: 1.35rem;
            color: #1c2833;
          }
          .scenario-banner p {
            margin: 0;
            color: #566573;
            font-size: 0.95rem;
          }

          /* Stage cards */
          div[data-testid="stVerticalBlock"] > div.stage-shell {
            border: 1px solid #e3e8ef;
            border-radius: 12px;
            padding: 0.25rem 0.5rem;
            margin-bottom: 0.25rem;
            background: #ffffff;
          }

          /* Sidebar polish */
          section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e6ebf2;
          }
          section[data-testid="stSidebar"] .sidebar-brand {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1a5276;
            margin-bottom: 0.15rem;
          }
          section[data-testid="stSidebar"] .sidebar-tagline {
            font-size: 0.78rem;
            color: #6b7c93;
            margin-bottom: 1.25rem;
            line-height: 1.4;
          }

          /* Hide default Streamlit header/footer clutter */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
