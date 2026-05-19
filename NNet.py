
#============================================================================= #type: ignore  # noqa E501
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
#
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT https://www.optitrack.com/about/legal/eula.html #type: ignore  # noqa E501
# AND/OR FOR DOWNLOAD WITH THE APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING, INSTALLING, ACTIVATING #type: ignore  # noqa E501
# AND/OR OTHERWISE USING THE SOFTWARE, YOU ARE AGREEING THAT YOU HAVE READ, AND THAT YOU AGREE TO COMPLY WITH AND ARE #type: ignore  # noqa E501
# BOUND BY, THE PLUGINS EULA AND ALL APPLICABLE LAWS AND REGULATIONS. IF YOU DO NOT AGREE TO BE BOUND BY THE PLUGINS #type: ignore  # noqa E501
# EULA, THEN YOU MAY NOT DOWNLOAD, INSTALL, ACTIVATE OR OTHERWISE USE THE SOFTWARE AND YOU MUST PROMPTLY DELETE OR #type: ignore  # noqa E501
# RETURN IT. IF YOU ARE DOWNLOADING, INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE ON BEHALF OF AN ENTITY, #type: ignore  # noqa E501
# THEN BY DOING SO YOU REPRESENT AND WARRANT THAT YOU HAVE THE APPROPRIATE AUTHORITY TO ACCEPT THE PLUGINS EULA ON #type: ignore  # noqa E501
# BEHALF OF SUCH ENTITY. See license file in root directory for additional governing terms and information. #type: ignore  # noqa E501
#============================================================================= #type: ignore  # noqa E501


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish
# a connection and receive data via that NatNet connection
# to decode it using the NatNetClientLibrary.

import sys
import time
from PythonClient.NatNetClient import NatNetClient
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

class NatNetDataProcessor:
    """
    Lightweight wrapper around NatNetClient.

    Responsibilities:
    - configure a NatNetClient (addresses, multicast)
    - receive rigid-body updates via a callback
    - store the latest pose for each rigid body in rb_dict
    - provide a thread-safe accessor get_rigid_body_data() that returns a
      small numpy array [x, y, yaw] per rigid body
    """
    def __init__(self, server="127.0.0.1", client="127.0.0.1", use_multicast=True, stream_type='d'):
        # Initialize NatNetClient
        self.server = server
        self.client = client
        self.use_multicast = use_multicast
        self.stream_type = stream_type
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(self.client)
        self.streaming_client.set_server_address(self.server)
        self.streaming_client.set_use_multicast(self.use_multicast)
        # Set up rigid body listener
        self.streaming_client.rigid_body_listener = self.receive_rigid_body_frame
        # Initialize rigid body data storage
        self.rb_dict = {}
        self.rb_lock = threading.Lock()

    def receive_rigid_body_frame(self, new_id, position, rotation):
        """
        Callback invoked by NatNetClient for each rigid body in a received frame.

        Parameters:
        - new_id: integer rigid-body identifier assigned by Motive
        - position: iterable-like of 3 floats (x, y, z) in meters
        - rotation: quaternion (qx, qy, qz, qw) as iterable-like

        """
        # Creating the rigid body data structure
        rb_data = {
            "position": np.array(position),
            "orientation": np.array(rotation)
            }
        # Store the rigid body data
        with self.rb_lock:
            self.rb_dict[new_id] = rb_data

    def start_streaming(self):
        """
        Start the NatNet client's networking loop.

        - Calls NatNetClient.run(stream_type) which should start background threads.
        - Sleeps briefly to allow connection negotiation.
        - Checks connected() and exits on failure (mirrors original behavior).
        """
        # Start the NatNet client
        started = self.streaming_client.run(self.stream_type)
        if not started:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        # Sleep briefly to allow connection negotiation.
        time.sleep(1)
        # Check if the client is connected
        if self.streaming_client.connected() is False:
            print("ERROR: Could not connect properly.  Check that Motive streaming is on.") #type: ignore  # noqa F501
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

    def stop_streaming(self):
        """
        Stop the NatNet client's networking loop.
        """
        try:
            self.streaming_client.shutdown()
        except Exception:
            pass
    
    def q2yaw(self, q, degrees=False):
        """
        Convert a quaternion into a yaw angle.

        Parameters:
        - q: iterable-like of 4 floats (qx, qy, qz, qw)
        - degrees: boolean flag to return the angle in degrees (default: False)

        Returns:
        - yaw: float yaw angle in radians (or degrees if requested)
        """
        q1, q2, q3, q0 = q
        yaw = -np.arctan2(2.0 * (q1 * q2 - q0 * q3), 2.0 * (q0 * q0 + q1 * q1) - 1.0)
        return np.degrees(yaw) if degrees else yaw
        
    
    def get_rigid_body_data(self, degrees=False):
        """
        Get the current rigid body data.

        Parameters:
        - degrees: boolean flag to return yaw angles in degrees (default: False)

        Returns:
        - results: dictionary mapping rigid body IDs to their [x, y, yaw] positions
        """
        with self.rb_lock:
            results = {
                k: np.array([v["position"][0], v["position"][1], self.q2yaw(v["orientation"], degrees=degrees)])
                for k, v in self.rb_dict.items()
            }
            return results

