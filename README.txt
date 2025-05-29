Checkmate: Design and Development of a Portable Chessboard Appliance with Computer Vision
=============================================

This guide will help you set up and run the Chess Recognition System using Python 3 and YOLOv8.

Requirements:
-------------
- Python 3.10 or newer
- pip3 (Python package manager)
- A webcam or camera for capturing chessboard images
- GPU (optional, but recommended for faster YOLOv8 inference)

Installation Steps:
-------------------

1. **Unzip the Source Code**
   - Extract `CheckMate_Source_Code.zip` to your preferred location.
   - Open a terminal or command prompt and navigate to the extracted folder:
     ```
     cd thesis-chess-main
     ```

2. **Create a Python Virtual Environment and Install Dependencies**
   - Create a virtual environment:
     ```
     python3 -m venv .venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```
       .venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```
       source .venv/bin/activate
       ```
   - Install all required Python packages:
     ```
     pip3 install -r requirements.txt
     ```

3. **Run the Application**
   - Start the chess recognition app by running:
     ```
     python3 main.py
     ```
   - The application will launch with a menu:
     - **New Game**: Start a new chess game and use the recognition features.
     - **View History**: Review previous games and moves.

Notes:
------
- Make sure your webcam is connected before running the application.
- If you encounter errors about missing packages, re-run `pip install -r requirements.txt`.
- For camera issues, check your device index or permissions.
- For YOLOv8 errors, ensure the weights file path is correct and compatible with your code.
