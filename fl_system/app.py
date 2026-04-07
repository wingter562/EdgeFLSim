import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import threading
import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from config import Config
from simulation.runner import run_with_callback

app = Flask(__name__)
latest_round_data = None
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=600,      # 增加到10分钟
    ping_interval=60       # 每分钟发一次心跳
)

simulation_status = {"running": False}

def run_simulation(config):
    def callback(round_data):
        global latest_round_data
        latest_round_data = round_data
        socketio.emit('round_update', round_data)
    try:
        print("Starting simulation thread")
        run_with_callback(config, callback)
        print("Simulation finished")
    finally:
        # 确保状态重置，无论仿真是否异常
        simulation_status["running"] = False
        socketio.emit('simulation_complete', {})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start():
    if simulation_status["running"]:
        return jsonify({"error": "Simulation already running"}), 400
    data = request.json
    config = Config()
    config.num_edge_servers = data.get('num_edge_servers', config.num_edge_servers)
    config.num_devices = data.get('num_devices', config.num_devices)
    config.num_rounds = data.get('num_rounds', config.num_rounds)
    config.batch_size = data.get('batch_size', config.batch_size)
    config.selection_strategy = data.get('selection_strategy', config.selection_strategy)
    config.aggregation_method = data.get('aggregation_method', config.aggregation_method)
    config.learning_rate = data.get('learning_rate', config.learning_rate)
    config.local_epochs = data.get('local_epochs', config.local_epochs)
    config.gpu_ratio = data.get('gpu_ratio', config.gpu_ratio)
    config.base_station_mode = data.get('base_station_mode', config.base_station_mode)
    config.model_name = data.get('model_name', config.model_name)
    config.dataset_name = data.get('dataset_name', config.dataset_name)
    simulation_status["running"] = True
    thread = threading.Thread(target=run_simulation, args=(config,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started"})

@socketio.on('connect')
def handle_connect():
    if latest_round_data is not None:
        emit('round_update', latest_round_data)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)