# Citation Fetcher

This project is a Flask-based web application that fetches, processes, and displays citation data. The application uses a BERT model to identify citations from response texts and displays them in a user-friendly table format on a webpage. Users can refresh the data and save the processed data to a JSON file.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Technologies Used](#technologies-used)
- [License](#license)

## Features
- Fetch citations data from a pre-defined API.
- Process the data using a BERT model to identify relevant citations.
- Display the processed data in a table format on a webpage.
- Refresh button to update the displayed data.
- Save button to save the processed data to a JSON file.

## Installation

### Prerequisites
- Python 3.6+
- pip (Python package installer)

### Steps
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/citation-fetcher.git
    cd citation-fetcher
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask application:**
    ```bash
    python app.py
    ```

5. **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

- **Refresh Data:** Click the "Refresh" button on the webpage to fetch and display the latest citation data.
- **Save Data:** Click the "Save" button to save the currently displayed data to a JSON file.

## Endpoints

- **GET /citations**
  - Fetches the first 10 citation data entries from the `data.json` file.

- **POST /save**
  - Processes the example input data and saves it to `data.json`.

- **GET /**
  - Serves the main webpage where citations are displayed in a table.

## Technologies Used

- **Backend:**
  - Flask
  - Transformers (HuggingFace BERT model)
  - Torch

- **Frontend:**
  - HTML
  - JavaScript

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
