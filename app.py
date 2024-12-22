from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
from enum import Enum
import io
import os
from datetime import datetime

app = Flask(__name__)


class FollowerType(Enum):
    ROLLER = "roller"
    KNIFE_EDGE = "knife_edge"
    FLAT_FACE = "flat_face"


class MotionType(Enum):
    SHM = "simple_harmonic"
    UNIFORM_VELOCITY = "uniform_velocity"
    UNIFORM_ACCELERATION = "uniform_acceleration"


def calculate_lift_motion(fraction, motion_type):
    """Calculate lift motion displacement"""
    if motion_type == MotionType.SHM:
        # Simple Harmonic Motion
        return (1 - np.cos(np.pi * fraction)) / 2

    elif motion_type == MotionType.UNIFORM_VELOCITY:
        # Uniform Velocity (Linear motion)
        return fraction

    elif motion_type == MotionType.UNIFORM_ACCELERATION:
        # Uniform Acceleration and Retardation
        if fraction <= 0.5:
            # Acceleration phase (0 to 0.5)
            return 2 * fraction * fraction
        else:
            # Retardation phase (0.5 to 1)
            return 1 - 2 * (1 - fraction) * (1 - fraction)
    return 0


def calculate_fall_motion(fraction, motion_type):
    """Calculate fall motion displacement"""
    if motion_type == MotionType.SHM:
        # Simple Harmonic Motion
        return (1 + np.cos(np.pi * fraction)) / 2

    elif motion_type == MotionType.UNIFORM_VELOCITY:
        # Uniform Velocity (Linear motion)
        return 1 - fraction

    elif motion_type == MotionType.UNIFORM_ACCELERATION:
        # Uniform Acceleration and Retardation
        if fraction <= 0.5:
            # Acceleration phase
            return 1 - 2 * fraction * fraction
        else:
            # Retardation phase
            return 2 * (1 - fraction) * (1 - fraction)
    return 0


def generate_cam_profile(
        follower_type=FollowerType.ROLLER,
        base_circle_radius=25,
        roller_radius=12.5,
        follower_width=30,
        lift=40,
        lift_angle=120,
        fall_angle=150,
        pre_lift_dwell_angle=30,
        offset=12.5,
        lift_motion=MotionType.SHM,
        fall_motion=MotionType.UNIFORM_VELOCITY,
        num_points=720):

    # Calculate the post-fall dwell angle
    total_motion_angle = lift_angle + fall_angle
    post_fall_dwell_angle = 360 - (pre_lift_dwell_angle + total_motion_angle)

    # Arrays to store points
    theta = np.linspace(0, 2*np.pi, num_points)
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    for i, angle in enumerate(theta):
        angle_deg = np.rad2deg(angle)

        # First dwell (pre-lift)
        if angle_deg <= pre_lift_dwell_angle:
            displacement = 0

        # Lift period
        elif angle_deg <= (pre_lift_dwell_angle + lift_angle):
            lift_fraction = (angle_deg - pre_lift_dwell_angle) / lift_angle
            displacement = lift * \
                calculate_lift_motion(lift_fraction, lift_motion)

        # Dwell at maximum lift
        elif angle_deg <= (pre_lift_dwell_angle + lift_angle + post_fall_dwell_angle):
            displacement = lift

        # Fall period
        else:
            fall_fraction = (angle_deg - (pre_lift_dwell_angle +
                             lift_angle + post_fall_dwell_angle)) / fall_angle
            displacement = lift * \
                calculate_fall_motion(fall_fraction, fall_motion)

        # Calculate basic profile coordinates
        r = base_circle_radius + displacement
        x[i] = r * np.cos(angle) + offset * np.sin(angle)
        y[i] = r * np.sin(angle) - offset * np.cos(angle)
        z[i] = 0

    # Apply follower-specific modifications
    if follower_type == FollowerType.ROLLER:
        for i in range(num_points):
            angle = theta[i]
            x[i] += roller_radius * np.cos(angle)
            y[i] += roller_radius * np.sin(angle)

    elif follower_type == FollowerType.KNIFE_EDGE:
        pass  # Basic profile is used for knife edge

    elif follower_type == FollowerType.FLAT_FACE:
        for i in range(num_points):
            angle = theta[i]
            tangent_angle = np.arctan2(y[i], x[i])
            x[i] += (follower_width/2) * np.sin(tangent_angle)
            y[i] -= (follower_width/2) * np.cos(tangent_angle)

    return x, y, z


def save_to_adams(x, y, z, filename='cam_profile.txt'):
    """Save the profile points in Adams View compatible format"""
    with open(filename, 'w') as f:
        for i in range(len(x)):
            f.write(f"{x[i]:.6f}, {y[i]:.6f}, {z[i]:.6f}\n")
    print(f"Profile saved to {filename}")


def save_to_memory(x, y, z):
    """Save the profile points to a memory buffer instead of a file"""
    buffer = io.StringIO()
    for i in range(len(x)):
        buffer.write(f"{x[i]:.6f}, {y[i]:.6f}, {z[i]:.6f}\n")
    buffer.seek(0)
    return buffer


@app.route('/')
def index():
    return render_template('index.html',
                           follower_types=[
                               type.value for type in FollowerType],
                           motion_types=[type.value for type in MotionType])


@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get form data
        data = request.form

        # Convert form data to appropriate types
        params = {
            'follower_type': FollowerType(data.get('follower_type')),
            'base_circle_radius': float(data.get('base_circle_radius', 25)),
            'roller_radius': float(data.get('roller_radius', 12.5)),
            'follower_width': float(data.get('follower_width', 30)),
            'lift': float(data.get('lift', 40)),
            'lift_angle': float(data.get('lift_angle', 120)),
            'fall_angle': float(data.get('fall_angle', 150)),
            'pre_lift_dwell_angle': float(data.get('pre_lift_dwell_angle', 30)),
            'offset': float(data.get('offset', 12.5)),
            'lift_motion': MotionType(data.get('lift_motion')),
            'fall_motion': MotionType(data.get('fall_motion')),
            'num_points': int(data.get('num_points', 720))
        }

        # Generate cam profile
        x, y, z = generate_cam_profile(**params)

        # Save to memory buffer
        buffer = save_to_memory(x, y, z)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cam_profile_{timestamp}.txt"

        # Send file
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run()
