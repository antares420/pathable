from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import random
import json
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'navigation_system_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Navigation system state
navigation_state = {
    'is_running': False,
    'mode': 'demo',  # 'demo', 'camera', or 'phone'
    'distance': 0,
    'direction': '',
    'last_instruction': '',
    'obstacle_detected': False,
    'camera_url': '',
    'camera_connected': False,
    'wheelchair_mode': True,
    'accessibility_obstacles': [],
    'surface_type': 'unknown',
    'path_width': 0,
    'slope_detected': False,
    'accessibility_signs': [],
    'alternative_route_suggestion': ''
}

# Camera configuration
camera_config = {
    'phone_ip': '',  # User's phone IP for camera streaming
    'usb_camera_id': 0,  # Default USB camera
    'current_camera': None
}

def analyze_camera_frame(frame):
    """Analyze camera frame for navigation guidance"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Split frame into left and right halves
    height, width = edges.shape
    left_half = edges[:, :width//2]
    right_half = edges[:, width//2:]
    
    # Count edge pixels (lower count = more open space)
    left_score = np.sum(left_half) / 255
    right_score = np.sum(right_half) / 255
    
    # Estimate distance based on bottom portion of frame
    bottom_portion = edges[int(height*0.7):, :]
    edge_density = np.sum(bottom_portion) / (bottom_portion.shape[0] * bottom_portion.shape[1] * 255)
    
    # Convert edge density to distance estimate (rough approximation)
    estimated_distance = max(20, 200 - (edge_density * 1000))
    
    # If wheelchair mode is enabled, add wheelchair-specific analysis
    basic_analysis = {
        'distance': int(estimated_distance),
        'direction': 'left' if left_score < right_score else 'right'
    }
    
    if navigation_state.get('wheelchair_mode', True):
        wheelchair_analysis = analyze_wheelchair_accessibility(frame, gray, edges)
        basic_analysis.update(wheelchair_analysis)
    
    return basic_analysis

def analyze_wheelchair_accessibility(frame, gray, edges):
    """Analyze frame specifically for wheelchair accessibility obstacles"""
    height, width = gray.shape
    accessibility_obstacles = []
    surface_type = 'unknown'
    path_width = 100  # Default assumption: wide enough
    slope_detected = False
    accessibility_signs = []
    
    # 1. Detect stairs using horizontal line detection
    stairs_detected = detect_stairs(edges, height, width)
    if stairs_detected:
        accessibility_obstacles.append({
            'type': 'stairs',
            'severity': 'critical',
            'message': 'STAIRS AHEAD - Find alternative route with ramp or elevator'
        })
    
    # 2. Detect curbs using edge detection in lower portion
    curb_detected = detect_curbs(edges, height, width)
    if curb_detected:
        accessibility_obstacles.append({
            'type': 'curb',
            'severity': 'high',
            'message': 'CURB detected - Look for curb cut or alternative path'
        })
    
    # 3. Analyze path width using edge detection
    path_width = estimate_path_width(edges, height, width)
    if path_width < 90:  # Less than wheelchair width requirement
        accessibility_obstacles.append({
            'type': 'narrow_path',
            'severity': 'high',
            'message': f'Path too narrow ({path_width}cm) - Find wider route'
        })
    
    # 4. Detect steep slopes using perspective analysis
    slope_detected = detect_slope(gray, height, width)
    if slope_detected:
        accessibility_obstacles.append({
            'type': 'steep_slope',
            'severity': 'high',
            'message': 'STEEP SLOPE detected - Proceed with caution or find gentler route'
        })
    
    # 5. Analyze surface texture for wheelchair suitability
    surface_type = analyze_surface_texture(gray)
    if surface_type in ['rough', 'gravel', 'sand']:
        accessibility_obstacles.append({
            'type': 'rough_surface',
            'severity': 'medium',
            'message': f'{surface_type.upper()} surface - May be difficult for wheelchair navigation'
        })
    
    # 6. Look for accessibility signs (simplified detection)
    accessibility_signs = detect_accessibility_signs(frame)
    
    return {
        'accessibility_obstacles': accessibility_obstacles,
        'surface_type': surface_type,
        'path_width': path_width,
        'slope_detected': slope_detected,
        'accessibility_signs': accessibility_signs
    }

def detect_stairs(edges, height, width):
    """Detect stairs using horizontal line detection"""
    # Look for multiple horizontal lines in the middle-bottom area
    middle_bottom = edges[int(height*0.4):int(height*0.9), :]
    
    # Use HoughLines to detect horizontal lines
    lines = cv2.HoughLines(middle_bottom, 1, np.pi/180, threshold=int(width*0.3))
    
    horizontal_lines = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Check if line is approximately horizontal (theta close to 0 or π)
            if abs(theta) < 0.3 or abs(theta - np.pi) < 0.3:
                horizontal_lines += 1
    
    # If we detect multiple horizontal lines, likely stairs
    return horizontal_lines >= 3

def detect_curbs(edges, height, width):
    """Detect curbs using edge detection in lower portion"""
    # Focus on the bottom 30% of the image
    bottom_portion = edges[int(height*0.7):, :]
    
    # Look for strong horizontal edges that might indicate curbs
    horizontal_edges = np.sum(bottom_portion, axis=1)
    
    # If there's a sudden spike in horizontal edges, might be a curb
    if len(horizontal_edges) > 0:
        max_edges = np.max(horizontal_edges)
        avg_edges = np.mean(horizontal_edges)
        return max_edges > avg_edges * 2.5
    
    return False

def estimate_path_width(edges, height, width):
    """Estimate path width based on edge detection"""
    # Analyze the middle portion of the image
    middle_section = edges[int(height*0.4):int(height*0.8), :]
    
    path_width = 100  # Default assumption
    
    # Find the leftmost and rightmost edges that might indicate path boundaries
    for row in range(middle_section.shape[0]):
        edge_row = middle_section[row, :]
        non_zero_indices = np.where(edge_row > 0)[0]
        
        if len(non_zero_indices) >= 2:
            left_edge = non_zero_indices[0]
            right_edge = non_zero_indices[-1]
            
            # Convert pixel width to approximate centimeters (rough estimation)
            pixel_width = right_edge - left_edge
            estimated_width = (pixel_width / width) * 200  # Assume 2m field of view
            path_width = max(30, min(200, estimated_width))
            break
    
    return int(path_width)

def detect_slope(gray, height, width):
    """Detect steep slopes using perspective analysis"""
    # Simple slope detection based on image perspective distortion
    top_half = gray[:int(height*0.5), :]
    bottom_half = gray[int(height*0.5):, :]
    
    # Calculate average brightness difference
    top_brightness = np.mean(top_half)
    bottom_brightness = np.mean(bottom_half)
    
    # Steep upward slope: bottom significantly darker than top
    # Steep downward slope: top significantly darker than bottom
    brightness_diff = abs(top_brightness - bottom_brightness)
    
    return brightness_diff > 40  # Threshold for slope detection

def analyze_surface_texture(gray):
    """Analyze surface texture for wheelchair suitability"""
    # Use image variance to estimate surface roughness
    variance = np.var(gray)
    
    if variance > 2000:
        return 'rough'
    elif variance > 1000:
        return 'textured'
    elif variance < 200:
        return 'smooth'
    else:
        return 'normal'

def detect_accessibility_signs(frame):
    """Detect accessibility-related signs (simplified detection)"""
    signs = []
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Look for blue color (common in accessibility signs)
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # If significant blue area detected, might be accessibility sign
    blue_pixels = np.sum(blue_mask > 0)
    total_pixels = frame.shape[0] * frame.shape[1]
    
    if blue_pixels > total_pixels * 0.02:  # More than 2% blue pixels
        signs.append({
            'type': 'possible_accessibility_sign',
            'color': 'blue',
            'message': 'Potential accessibility signage detected'
        })
    
    return signs

def generate_wheelchair_instruction(analysis):
    """Generate wheelchair-specific navigation instruction"""
    obstacles = analysis.get('accessibility_obstacles', [])
    distance = analysis.get('distance', 100)
    direction = analysis.get('direction', 'straight')
    path_width = analysis.get('path_width', 100)
    surface_type = analysis.get('surface_type', 'unknown')
    
    # Priority: Critical obstacles first
    critical_obstacles = [obs for obs in obstacles if obs['severity'] == 'critical']
    if critical_obstacles:
        return critical_obstacles[0]['message']
    
    # High priority obstacles
    high_obstacles = [obs for obs in obstacles if obs['severity'] == 'high']
    if high_obstacles:
        return high_obstacles[0]['message']
    
    # Medium priority obstacles
    medium_obstacles = [obs for obs in obstacles if obs['severity'] == 'medium']
    if medium_obstacles:
        return medium_obstacles[0]['message']
    
    # Normal navigation guidance
    if distance < 50:
        return f"Obstacle {distance}cm ahead - Turn {direction}. Path width: {path_width}cm"
    elif path_width < 100:
        return f"Narrow path ahead ({path_width}cm) - Proceed carefully"
    elif surface_type != 'normal' and surface_type != 'unknown':
        return f"Path clear - {surface_type.upper()} surface ahead"
    else:
        return f"Path clear - Safe to continue straight"

def generate_route_suggestion(analysis):
    """Generate alternative route suggestion when obstacles are detected"""
    obstacles = analysis.get('accessibility_obstacles', [])
    
    for obstacle in obstacles:
        obstacle_type = obstacle['type']
        
        if obstacle_type == 'stairs':
            return "Look for: Elevator, ramp, or accessible entrance nearby"
        elif obstacle_type == 'curb':
            return "Look for: Curb cut, crosswalk, or accessible path"
        elif obstacle_type == 'narrow_path':
            return "Look for: Wider path, alternative route, or turn around space"
        elif obstacle_type == 'steep_slope':
            return "Look for: Gentler slope, elevator, or alternative level route"
        elif obstacle_type == 'rough_surface':
            return "Look for: Smoother paved path or indoor route"
    
    return ""

def navigation_with_camera():
    """Run navigation with real camera input"""
    cap = camera_config['current_camera']
    
    while navigation_state['is_running'] and cap is not None:
        ret, frame = cap.read()
        if not ret:
            navigation_state['camera_connected'] = False
            navigation_state['last_instruction'] = 'Camera connection lost - switching to demo mode'
            socketio.emit('navigation_update', navigation_state)
            # Fall back to simulation if camera fails
            navigation_simulation()
            break
            
        # Analyze frame for navigation
        analysis = analyze_camera_frame(frame)
        
        # Update state with real camera data
        navigation_state['distance'] = analysis['distance']
        navigation_state['direction'] = analysis['direction']
        navigation_state['obstacle_detected'] = analysis['distance'] < 50
        navigation_state['camera_connected'] = True
        navigation_state['mode'] = 'camera'
        
        # Update wheelchair-specific state
        if navigation_state.get('wheelchair_mode', True):
            navigation_state['accessibility_obstacles'] = analysis.get('accessibility_obstacles', [])
            navigation_state['surface_type'] = analysis.get('surface_type', 'unknown')
            navigation_state['path_width'] = analysis.get('path_width', 100)
            navigation_state['slope_detected'] = analysis.get('slope_detected', False)
            navigation_state['accessibility_signs'] = analysis.get('accessibility_signs', [])
            
            # Generate wheelchair-specific instruction
            instruction = generate_wheelchair_instruction(analysis)
            navigation_state['last_instruction'] = instruction
            navigation_state['alternative_route_suggestion'] = generate_route_suggestion(analysis)
        else:
            if analysis['distance'] < 50:
                instruction = f"Turn {analysis['direction']} now"
                navigation_state['last_instruction'] = instruction
            else:
                navigation_state['last_instruction'] = "Path is clear"
        
        # Emit real-time update to frontend
        socketio.emit('navigation_update', navigation_state)
        
        time.sleep(0.1)  # Faster updates with camera

def navigation_enhanced_simulation():
    """Enhanced simulation that mimics camera behavior with wheelchair obstacles"""
    while navigation_state['is_running']:
        # More realistic simulation that varies based on time
        import math
        current_time = time.time()
        
        # Create realistic movement patterns
        base_distance = 100 + 50 * math.sin(current_time * 0.5)
        noise = random.randint(-20, 20)
        distance = max(15, int(base_distance + noise))
        
        # More intelligent direction changes
        if distance < 60:
            # When close to obstacles, prefer turning away from previous direction
            if hasattr(navigation_state, 'last_turn'):
                direction = 'right' if navigation_state.get('last_turn') == 'left' else 'left'
            else:
                direction = random.choice(['left', 'right'])
            navigation_state['last_turn'] = direction
        else:
            direction = random.choice(['left', 'right', 'straight'])
        
        # Update basic state
        navigation_state['distance'] = distance
        navigation_state['direction'] = direction
        navigation_state['obstacle_detected'] = distance < 50
        navigation_state['mode'] = 'enhanced_demo'
        
        # Simulate wheelchair-specific obstacles
        if navigation_state.get('wheelchair_mode', True):
            simulate_wheelchair_obstacles(current_time)
            
            # Generate wheelchair-specific instruction
            mock_analysis = {
                'distance': distance,
                'direction': direction,
                'accessibility_obstacles': navigation_state['accessibility_obstacles'],
                'surface_type': navigation_state['surface_type'],
                'path_width': navigation_state['path_width'],
                'slope_detected': navigation_state['slope_detected']
            }
            instruction = generate_wheelchair_instruction(mock_analysis)
            navigation_state['last_instruction'] = instruction
            navigation_state['alternative_route_suggestion'] = generate_route_suggestion(mock_analysis)
        else:
            if distance < 50:
                instruction = f"Turn {direction} now - obstacle at {distance}cm"
                navigation_state['last_instruction'] = instruction
            elif distance < 80:
                instruction = f"Caution - obstacle ahead at {distance}cm"
                navigation_state['last_instruction'] = instruction
            else:
                navigation_state['last_instruction'] = "Path is clear - safe to continue"
        
        # Emit real-time update to frontend
        socketio.emit('navigation_update', navigation_state)
        
        time.sleep(1.5)  # Balanced update speed

def simulate_wheelchair_obstacles(current_time):
    """Simulate various wheelchair accessibility obstacles for demo"""
    accessibility_obstacles = []
    
    # Simulate different obstacles based on time cycles
    time_factor = current_time % 60  # 60-second cycle
    
    if 0 <= time_factor < 10:
        # Simulate stairs
        if random.random() < 0.3:
            accessibility_obstacles.append({
                'type': 'stairs',
                'severity': 'critical',
                'message': 'STAIRS AHEAD - Find alternative route with ramp or elevator'
            })
        navigation_state['surface_type'] = 'normal'
        navigation_state['path_width'] = random.randint(120, 200)
        navigation_state['slope_detected'] = False
        
    elif 10 <= time_factor < 20:
        # Simulate curbs
        if random.random() < 0.4:
            accessibility_obstacles.append({
                'type': 'curb',
                'severity': 'high',
                'message': 'CURB detected - Look for curb cut or alternative path'
            })
        navigation_state['surface_type'] = 'normal'
        navigation_state['path_width'] = random.randint(100, 160)
        navigation_state['slope_detected'] = False
        
    elif 20 <= time_factor < 30:
        # Simulate narrow paths
        path_width = random.randint(60, 110)
        if path_width < 90:
            accessibility_obstacles.append({
                'type': 'narrow_path',
                'severity': 'high',
                'message': f'Path too narrow ({path_width}cm) - Find wider route'
            })
        navigation_state['surface_type'] = 'normal'
        navigation_state['path_width'] = path_width
        navigation_state['slope_detected'] = False
        
    elif 30 <= time_factor < 40:
        # Simulate steep slopes
        if random.random() < 0.5:
            accessibility_obstacles.append({
                'type': 'steep_slope',
                'severity': 'high',
                'message': 'STEEP SLOPE detected - Proceed with caution or find gentler route'
            })
            navigation_state['slope_detected'] = True
        navigation_state['surface_type'] = 'normal'
        navigation_state['path_width'] = random.randint(120, 200)
        
    elif 40 <= time_factor < 50:
        # Simulate rough surfaces
        surface_types = ['rough', 'gravel', 'textured', 'normal']
        surface = random.choice(surface_types)
        if surface in ['rough', 'gravel']:
            accessibility_obstacles.append({
                'type': 'rough_surface',
                'severity': 'medium',
                'message': f'{surface.upper()} surface - May be difficult for wheelchair navigation'
            })
        navigation_state['surface_type'] = surface
        navigation_state['path_width'] = random.randint(120, 200)
        navigation_state['slope_detected'] = False
        
    else:
        # Clear path simulation
        navigation_state['surface_type'] = 'normal'
        navigation_state['path_width'] = random.randint(120, 200)
        navigation_state['slope_detected'] = False
    
    # Simulate accessibility signs occasionally
    accessibility_signs = []
    if random.random() < 0.1:
        accessibility_signs.append({
            'type': 'possible_accessibility_sign',
            'color': 'blue',
            'message': 'Potential accessibility signage detected'
        })
    
    navigation_state['accessibility_obstacles'] = accessibility_obstacles
    navigation_state['accessibility_signs'] = accessibility_signs

def navigation_simulation():
    """Simulate the navigation system for demo purposes"""
    while navigation_state['is_running']:
        # Generate random distance and direction
        distance = random.randint(10, 200)
        direction = random.choice(['left', 'right'])
        
        # Update state
        navigation_state['distance'] = distance
        navigation_state['direction'] = direction
        navigation_state['obstacle_detected'] = distance < 50
        
        if distance < 50:
            instruction = f"Turn {direction} now"
            navigation_state['last_instruction'] = instruction
        else:
            navigation_state['last_instruction'] = "Path is clear"
        
        # Emit real-time update to frontend
        socketio.emit('navigation_update', navigation_state)
        
        time.sleep(2)  # Update every 2 seconds

@app.route('/')
def index():
    # Detect mobile devices and redirect to mobile interface
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_keywords = ['mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone']
    
    if any(keyword in user_agent for keyword in mobile_keywords):
        return render_template('mobile.html')
    else:
        return render_template('index.html')

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/maps')
def maps():
    return render_template('maps.html')

@app.route('/api/status')
def get_status():
    return jsonify(navigation_state)

@app.route('/api/baato-key')
def get_baato_key():
    import os
    api_key = os.environ.get('BAATO_API_KEY')
    if api_key:
        return jsonify({'api_key': api_key})
    else:
        return jsonify({'error': 'API key not configured'}), 404

@app.route('/api/openroute-key')
def get_openroute_key():
    import os
    api_key = os.environ.get('OPENROUTESERVICE_API_KEY')
    if api_key:
        return jsonify({'api_key': api_key})
    else:
        return jsonify({'error': 'OpenRouteService API key not configured'}), 404

@app.route('/api/camera/connect', methods=['POST'])
def connect_camera():
    data = request.get_json()
    camera_type = data.get('type', 'usb')  # 'usb' or 'phone'
    
    try:
        if camera_type == 'phone':
            phone_ip = data.get('ip', '')
            if not phone_ip:
                return jsonify({'success': False, 'message': 'Phone IP required'})
            
            # Try connecting to phone camera (DroidCam format)
            camera_url = f"http://{phone_ip}:4747/video"
            cap = cv2.VideoCapture(camera_url)
            
            if cap.isOpened():
                camera_config['current_camera'] = cap
                camera_config['phone_ip'] = phone_ip
                navigation_state['camera_url'] = camera_url
                navigation_state['camera_connected'] = True
                return jsonify({'success': True, 'message': f'Connected to phone camera at {phone_ip}'})
            else:
                return jsonify({'success': False, 'message': 'Could not connect to phone camera'})
                
        elif camera_type == 'usb':
            camera_id = data.get('id', 0)
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                camera_config['current_camera'] = cap
                camera_config['usb_camera_id'] = camera_id
                navigation_state['camera_connected'] = True
                return jsonify({'success': True, 'message': f'Connected to USB camera {camera_id}'})
            else:
                return jsonify({'success': False, 'message': f'Could not connect to USB camera {camera_id}'})
                
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error connecting camera: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Unknown camera type'})

@app.route('/api/camera/disconnect', methods=['POST'])
def disconnect_camera():
    if camera_config['current_camera'] is not None:
        camera_config['current_camera'].release()
        camera_config['current_camera'] = None
        navigation_state['camera_connected'] = False
        navigation_state['camera_url'] = ''
        return jsonify({'success': True, 'message': 'Camera disconnected'})
    return jsonify({'success': False, 'message': 'No camera connected'})

@app.route('/api/camera/status')
def camera_status():
    return jsonify({
        'connected': navigation_state['camera_connected'],
        'url': navigation_state['camera_url'],
        'phone_ip': camera_config['phone_ip'],
        'usb_id': camera_config['usb_camera_id']
    })

@socketio.on('start_navigation')
def handle_start_navigation(data=None):
    navigation_state['is_running'] = True
    
    # Check if camera is available
    if camera_config['current_camera'] is not None:
        navigation_state['mode'] = 'camera'
        thread = threading.Thread(target=navigation_with_camera)
    else:
        navigation_state['mode'] = 'enhanced_demo'
        thread = threading.Thread(target=navigation_enhanced_simulation)
    
    thread.daemon = True
    thread.start()
    
    emit('navigation_started', {
        'message': f'Navigation started in {navigation_state["mode"]} mode',
        'mode': navigation_state['mode']
    })

@socketio.on('stop_navigation')
def handle_stop_navigation():
    navigation_state['is_running'] = False
    navigation_state['last_instruction'] = 'Navigation stopped'
    emit('navigation_stopped', {'message': 'Navigation system stopped'})

@socketio.on('connect_phone_camera')
def handle_connect_phone_camera(data):
    phone_ip = data.get('ip', '')
    if not phone_ip:
        emit('camera_error', {'message': 'Please provide phone IP address'})
        return
        
    try:
        camera_url = f"http://{phone_ip}:4747/video"
        cap = cv2.VideoCapture(camera_url)
        
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret:
                camera_config['current_camera'] = cap
                camera_config['phone_ip'] = phone_ip
                navigation_state['camera_url'] = camera_url
                navigation_state['camera_connected'] = True
                emit('camera_connected', {
                    'message': f'Successfully connected to phone camera',
                    'ip': phone_ip
                })
            else:
                cap.release()
                emit('camera_error', {'message': 'Camera connected but no video stream available'})
        else:
            emit('camera_error', {'message': 'Could not connect to phone camera. Check IP and DroidCam app.'})
            
    except Exception as e:
        emit('camera_error', {'message': f'Connection error: {str(e)}'})

@socketio.on('connect_usb_camera')
def handle_connect_usb_camera(data):
    camera_id = data.get('id', 0)
    
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_config['current_camera'] = cap
                camera_config['usb_camera_id'] = camera_id
                navigation_state['camera_connected'] = True
                emit('camera_connected', {
                    'message': f'Successfully connected to USB camera {camera_id}'
                })
            else:
                cap.release()
                emit('camera_error', {'message': f'USB camera {camera_id} connected but no video available'})
        else:
            emit('camera_error', {'message': f'Could not connect to USB camera {camera_id}'})
            
    except Exception as e:
        emit('camera_error', {'message': f'USB camera error: {str(e)}'})

@socketio.on('camera_analysis')
def handle_camera_analysis(data):
    """Handle real-time camera analysis from mobile client"""
    if navigation_state['is_running']:
        # Update navigation state with mobile analysis
        navigation_state['distance'] = data.get('distance', 0)
        navigation_state['direction'] = data.get('direction', 'straight')
        navigation_state['obstacle_detected'] = data.get('distance', 999) < 50
        navigation_state['mode'] = 'mobile_camera'
        
        if navigation_state['obstacle_detected']:
            instruction = f"Turn {navigation_state['direction']} now!"
            navigation_state['last_instruction'] = instruction
        else:
            navigation_state['last_instruction'] = "Path is clear"
        
        # Broadcast update to all connected clients
        socketio.emit('navigation_update', navigation_state)

@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to navigation system'})

if __name__ == '__main__':
    # Clean up cameras on exit
    import atexit
    def cleanup():
        if camera_config['current_camera'] is not None:
            camera_config['current_camera'].release()
    atexit.register(cleanup)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=True, log_output=True)