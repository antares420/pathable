# Wheelchair Navigation Assistant

## Overview

A comprehensive real-time accessibility-focused navigation system specifically designed for wheelchair users. The application combines Flask backend with WebSocket communication, advanced computer vision for wheelchair-specific obstacle detection, and text-to-speech functionality to provide audio navigation guidance. The system detects wheelchair accessibility obstacles like stairs, curbs, narrow passages, steep slopes, and surface types while providing alternative route suggestions. It operates in demo mode (simulation) and camera mode (real-time analysis) to help wheelchair users navigate safely and find accessible routes.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

* **Localized Emergency Contact (September 5, 2025)**: Updated emergency contact number from 911 to 100 for Nepal's local emergency services
* **Enhanced Nepal Maps Interface (September 5, 2025)**: Complete Google Maps-like navigation interface using Baato API for Nepal with improved wheelchair-accessible route planning:
  - **Improved Location Selection**: Click-to-select functionality - users can click on input fields and then click anywhere on the map to select start and destination points with automatic reverse geocoding
  - **Real Road Routing**: Enhanced Baato API integration that follows actual roads and paths instead of showing straight lines between points
  - **Accessibility Obstacle Markers**: Red warning icons automatically placed on routes to mark stairs, steep slopes, and narrow paths that may be difficult for wheelchair users
  - **Enhanced Search**: Users can type location names and press Enter to search and automatically place markers
  - **Custom Markers**: Beautiful custom icons for start points (🚀), destinations (🏁), and various obstacle types (🚫 for stairs, ⛰️ for slopes, ↔️ for narrow paths)
* **Dual Navigation System**: Two distinct interfaces - mobile camera navigation for real-time obstacle detection and desktop maps navigation for route planning
* **Wheelchair Route Optimization**: Advanced routing with options to avoid stairs, prefer low traffic routes, smooth surfaces, and accessible entrances
* **Baato API Integration**: Full integration with Nepal's mapping service for accurate local navigation, search, and place discovery
* **Accessibility-First Design**: All interfaces optimized for wheelchair users with large touch targets, high contrast, and voice guidance

## System Architecture

### Frontend Architecture
- **Dual Interface System**: 
  - **Mobile Camera Interface**: Real-time wheelchair obstacle detection with camera integration
  - **Desktop Maps Interface**: Google Maps-like navigation using Baato API for Nepal
- **WebSocket Client**: Real-time bidirectional communication using Socket.IO for instant navigation updates
- **Audio Feedback System**: Text-to-speech integration for spoken wheelchair-specific navigation instructions and obstacle alerts
- **Interactive Maps**: Leaflet-based mapping with custom wheelchair accessibility markers and route visualization
- **Search & Discovery**: Location search, nearby accessible places, and turn-by-turn navigation
- **Responsive Design**: Grid-based layout with high-contrast elements and wheelchair accessibility information

### Backend Architecture
- **Flask Web Server**: Lightweight Python web framework serving the main application
- **WebSocket Server**: Flask-SocketIO implementation for real-time communication between frontend and backend
- **Wheelchair Navigation State Management**: Centralized state object tracking system status, distance measurements, direction guidance, wheelchair accessibility obstacles, path width, surface types, and alternative route suggestions
- **Wheelchair-Aware Simulation Engine**: Threading-based background process generating realistic wheelchair accessibility scenarios including stairs, curbs, slopes, and surface conditions

### Computer Vision Pipeline
- **MiDaS Depth Estimation**: Intel's deep learning model (DPT_Large variant) for monocular depth estimation
- **PyTorch Integration**: GPU-accelerated inference when CUDA is available, falling back to CPU processing
- **Camera Input Processing**: OpenCV-based video capture supporting IP camera streams (e.g., smartphone cameras via DroidCam)
- **Wheelchair Accessibility Analysis**: Specialized frame analysis detecting stairs, curbs, narrow passages, steep slopes, and surface textures
- **Path Width Estimation**: Real-time analysis to ensure paths are wide enough for wheelchair navigation (minimum 90cm)
- **Surface Type Detection**: Analysis of ground texture to identify wheelchair-suitable surfaces (smooth, textured, rough, gravel)
- **Accessibility Sign Recognition**: Detection of blue accessibility signage and wayfinding elements

### Audio System
- **Text-to-Speech Engine**: pyttsx3 library providing configurable speech synthesis with adjustable rate and voice selection
- **Wheelchair-Specific Audio Feedback**: Immediate spoken instructions for wheelchair accessibility obstacles, path width warnings, surface condition alerts, and alternative route suggestions
- **Priority-Based Voice Alerts**: Critical accessibility obstacles (stairs) receive immediate voice alerts with enhanced vibration patterns
- **Alternative Route Guidance**: Spoken suggestions for finding ramps, elevators, curb cuts, and accessible entrances

### Communication Architecture
- **RESTful API Endpoints**: Standard HTTP routes for status checking and system state queries
- **WebSocket Events**: Real-time event broadcasting for navigation updates, system status changes, and alert notifications
- **Cross-Origin Resource Sharing**: CORS-enabled WebSocket connections supporting diverse client environments

## External Dependencies

### Core Frameworks
- **Flask**: Python web framework for HTTP server and routing
- **Flask-SocketIO**: WebSocket implementation for real-time communication
- **OpenCV (cv2)**: Computer vision library for camera input and image processing

### Machine Learning & AI
- **PyTorch**: Deep learning framework for MiDaS model execution
- **Intel MiDaS Models**: Pre-trained depth estimation models loaded via torch.hub
- **NumPy**: Numerical computing library for array operations and depth map processing

### Mapping & Navigation APIs
- **Baato API**: Nepal's comprehensive mapping service for location search, routing, and place discovery
- **Leaflet**: Open-source JavaScript mapping library for interactive map interfaces
- **OpenStreetMap**: Base map tiles for comprehensive Nepal coverage

### Audio Processing
- **pyttsx3**: Cross-platform text-to-speech synthesis library

### Frontend Libraries
- **Socket.IO Client**: JavaScript WebSocket client library for real-time communication

### Hardware Integrations
- **IP Camera Streams**: Support for smartphone cameras via network streaming (DroidCam or similar applications)
- **CUDA Support**: Optional GPU acceleration for depth estimation when available