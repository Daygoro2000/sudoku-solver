import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime

# Importaciones de tus nuevos módulos
from vision import preprocess, main_outline, reframe, splitcells, crop_cell
from model import load_sudoku_model, read_cells
from solver import solve_with_timeout

# Obtener la ruta del directorio raíz (un nivel arriba de src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Carga del modelo inicializada de forma segura
model = load_sudoku_model('models/modelo_digitos.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    original_board = None
    
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # 1. Visión Artificial
            sudoku_img = cv2.resize(cv2.imread(filepath), (450, 450))
            contour, _ = cv2.findContours(preprocess(sudoku_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            biggest, _ = main_outline(contour)
            
            if biggest.size != 0:
                pts1 = np.float32(reframe(biggest))
                matrix = cv2.getPerspectiveTransform(pts1, np.float32([[0,0],[450,0],[0,450],[450,450]]))
                image_wrap = cv2.cvtColor(cv2.warpPerspective(sudoku_img, matrix, (450,450)), cv2.COLOR_BGR2GRAY)
                
                # 2. Inferencia (Deep Learning)
                grid = read_cells(crop_cell(splitcells(image_wrap)), model)
                grid_matrix = np.reshape(grid, (9,9)).tolist()
                original_board = [row[:] for row in grid_matrix]
                
                # 3. Resolución (Backtracking)
                if solve_with_timeout(grid_matrix, timeout=5):
                    prediction = grid_matrix
    
    return render_template('index.html', prediction=prediction, original_board=original_board, current_year=datetime.now().year)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)