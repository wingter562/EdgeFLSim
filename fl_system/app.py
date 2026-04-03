import os
import threading
import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from config import Config
from simulation.runner import run_with_callback

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=300, ping_interval=30)

simulation_status = {"running": False}

def run_simulation(config):
    def callback(round_data):
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

    simulation_status["running"] = True
    thread = threading.Thread(target=run_simulation, args=(config,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started"})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)