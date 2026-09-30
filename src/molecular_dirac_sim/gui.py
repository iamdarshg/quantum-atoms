"""Launch the local Streamlit GUI."""
import sys
from pathlib import Path


def main():
    try:
        from streamlit.web import cli
    except ImportError as exc:
        raise SystemExit("Install the GUI extra first: pip install -e '.[gui]'") from exc
    sys.argv=["streamlit","run",str(Path(__file__).with_name("app.py"))]
    cli.main()


if __name__=="__main__": main()
