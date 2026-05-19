import time
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.time_sync import TimeSyncThread
from bosdyn.client.robot_command import blocking_stand
from bosdyn.util import RobotTimeConverter


class SDKSpot:
    """Spot SDK wrapper with asynchronous velocity commands and automatic lease keep-alive."""

    def __init__(self, hostname, username, password):
        self.hostname = hostname
        self.username = username
        self.password = password

        self.sdk = None
        self.robot = None
        self.command_client = None
        self.lease_client = None
        self.lease = None
        self.lease_keep_alive = None
        self.time_sync = None
        self.time_converter = None  # Add this

    # ------------------------------------------------------------
    # Connect & Initialize
    # ------------------------------------------------------------
    def connect(self):
        """Connect to Spot, authenticate, acquire lease, start time sync and keep-alive."""
        print(f"[INFO] Connecting to Spot at {self.hostname}...")

        # Create SDK and robot object
        self.sdk = create_standard_sdk("SDKSpot_Client")
        self.robot = self.sdk.create_robot(self.hostname)

        # Authenticate
        self.robot.authenticate(self.username, self.password)

        # Start time sync
        # self.time_sync = TimeSyncThread(self.robot)
        # self.time_sync.start()
        self.robot.sync_with_directory()
        self.robot.time_sync.wait_for_sync(timeout_sec=10)
        print("[INFO] Time synchronization established.")

        # Create time converter
        self.time_converter = RobotTimeConverter(0)  

        # Acquire lease
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.lease = self.lease_client.acquire()
        print("[INFO] Lease acquired.")

        # --- Lease Keep-Alive: Correct implementation ---
        # Pass the lease object to the keep-alive. It starts automatically.
        self.lease_keep_alive = LeaseKeepAlive(self.lease_client)
        print("[INFO] Lease keep-alive started.")

        # Power on and command client setup
        self.robot.power_on(timeout_sec=20)
        print("[INFO] Spot motors powered on.")

        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        # Stand Spot
        blocking_stand(self.command_client, timeout_sec=10)
        print("[INFO] Spot is standing and ready.")

    # ------------------------------------------------------------
    # Send Asynchronous Velocity Command
    # ------------------------------------------------------------
    def send_velocity_command(self, v_x, v_y, v_rot, duration=5):
        """
        Send an asynchronous velocity command to Spot.
        v_x, v_y, v_rot: velocities in m/s and rad/s (body frame)
        duration: how long (seconds) the command should remain valid
        """
        if not self.command_client:
            raise RuntimeError("Robot not connected. Call connect() first.")
        # robot_timestamp = self.robot.time_sync.robot_timestamp_from_local_secs(time.time())
        # current_robot_time = robot_timestamp.seconds + robot_timestamp.nanos*1e-9
        current_robot_time = self.time_converter.robot_seconds_from_local_seconds(time.time())  # Fix: use instance method
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x,
            v_y=v_y,
            v_rot=v_rot,
            params=RobotCommandBuilder.mobility_params(stair_hint=False)
        )
        self.command_client.robot_command_async(cmd, end_time_secs=current_robot_time + duration)
        print(f"[CMD] Sent velocity command vx={v_x:.2f}, vy={v_y:.2f}, vrot={v_rot:.2f}")

    # ------------------------------------------------------------
    # Stop & Disconnect
    # ------------------------------------------------------------
    def stop_and_disconnect(self):
        power_down = RobotCommandBuilder.safe_power_off_command()
        self.command_client.robot_command(power_down)
        print("[INFO] Power down command sent.")

        self.lease_keep_alive.shutdown()
        print("[INFO] Lease keep-alive stopped.")

        self.lease_client.return_lease(self.lease)
        print("[INFO] Lease returned.")

        print("[INFO] Disconnected from Spot.")