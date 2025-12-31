#!/usr/bin/env python3
"""
LAUNCHER v2 - Start AI Server, wait for READY, then launch Main GUI
"""

import subprocess
import time
import sys
import threading
from pathlib import Path


class Launcher:
    def __init__(self):
        self.server_process = None
        self.gui_process = None
        self.is_server_ready = False
    
    def start_ai_server(self):
        """Start AI server (with GUI if available)"""
        print("\n" + "=" * 70)
        print("🚀 STEP 1: Starting AI Server with Real-time Monitoring")
        print("=" * 70)
        
        server_file = Path("ai_server_with_gui.py")
        if not server_file.exists():
            print("❌ Error: ai_server_with_gui.py not found!")
            return False
        
        print(f"📡 Launching: python {server_file}")
        print("-" * 70)
        
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=1,
            )
            return True
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def wait_for_server_ready(self, timeout=15):
        """Wait for server to signal READY"""
        print("\n⏳ Waiting for AI Server to be READY...")
        print("-" * 70)
        
        start_time = time.time()
        ready_marker = "[AIServer] ✅ READY"
        
        def read_server_output():
            try:
                while True:
                    line = self.server_process.stdout.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        print(line_str)
                        
                        # Check for ready signal
                        if ready_marker in line_str:
                            self.is_server_ready = True
                            return True
            except:
                pass
            return False
        
        # Start reader thread
        reader_thread = threading.Thread(target=read_server_output, daemon=True)
        reader_thread.start()
        
        # Wait for ready or timeout
        while time.time() - start_time < timeout:
            if self.is_server_ready:
                print("-" * 70)
                print("✅ AI SERVER READY!")
                return True
            time.sleep(0.1)
        
        print("-" * 70)
        print("❌ Timeout waiting for server!")
        return False
    
    def start_main_gui(self):
        """Start main data consolidation GUI"""
        print("\n" + "=" * 70)
        print("🎨 STEP 2: Launching Data Consolidation GUI")
        print("=" * 70)
        
        gui_file = Path("consolidation_gui_fixed_v6.py")
        if not gui_file.exists():
            print("❌ Error: consolidation_gui_fixed_v6.py not found!")
            return False
        
        print(f"📊 Starting: python {gui_file}")
        print("-" * 70)
        print("\n💡 The AI Server is running in the background.")
        print("💡 Graphs and metrics will appear in the AI Server window.")
        print("💡 Use the main GUI to select files and start processing.\n")
        
        try:
            self.gui_process = subprocess.Popen(
                [sys.executable, str(gui_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except Exception as e:
            print(f"❌ Failed to start GUI: {e}")
            return False
    
    def run(self):
        """Main launcher flow"""
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + " 🚀  DATA CONSOLIDATION + AI OPTIMIZATION SERVER  🚀 ".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "=" * 68 + "╝")
        
        # Step 1: Start server
        if not self.start_ai_server():
            return False
        
        # Step 2: Wait for server to be ready
        if not self.wait_for_server_ready():
            if self.server_process:
                self.server_process.terminate()
            return False
        
        # Give server time to fully initialize GUI
        print("\n⏳ Initializing server GUI...")
        time.sleep(2)
        
        # Step 3: Start main GUI
        if not self.start_main_gui():
            if self.server_process:
                self.server_process.terminate()
            return False
        
        print("\n" + "=" * 70)
        print("✅ EVERYTHING RUNNING!")
        print("=" * 70)
        print("\nWindow Layout:")
        print("  • AI Server Window:   Real-time graphs and metrics")
        print("  • Main GUI Window:    Data consolidation controls")
        print("\nTo Stop:")
        print("  • Close the Main GUI window (AI Server will stop automatically)")
        print("  • Or press Ctrl+C here\n")
        
        # Wait for GUI to close
        try:
            self.gui_process.wait()
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        
        # Cleanup
        self.cleanup()
        return True
    
    def cleanup(self):
        """Stop all processes"""
        print("\n⏹️  Shutting down...")
        
        if self.gui_process:
            try:
                self.gui_process.terminate()
                self.gui_process.wait(timeout=3)
            except:
                self.gui_process.kill()
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=3)
            except:
                self.server_process.kill()
        
        print("✅ Shutdown complete")


def main():
    launcher = Launcher()
    try:
        success = launcher.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        launcher.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        launcher.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()