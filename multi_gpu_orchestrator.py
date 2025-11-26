"""
Ultimate Bitcoin Puzzle Solver - Multi-GPU Orchestrator
Combines BitCrack + VanitySearch + keyhunt optimizations

Features:
- Multi-GPU work distribution
- Checkpoint/resume
- Real-time monitoring
- Email notifications
- Bloom filter optimization
"""

import subprocess
import json
import os
import time
import argparse
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage

# Email configuration
EMAIL_ADDRESS = "info@hermextravels.com"
SMTP_PASS = "wSsVR60krh6iCPwumDb7J7o4nl4BAl6kHUwvjFb37iP0G/zLpcdonkPLVA71GfZMR2BsR2EU8O0gzR8H0TEGiI8tww4JCyiF9mqRe1U4J3x17qnvhDzJV2pfkRCMLIMOww1rnGdjF8pu"
TO_ADDRESS = "praise.ordu@hermextravels.com"
SMTP_SERVER = "smtp.zeptomail.com"
SMTP_PORT = 587
SMTP_USER = "emailapikey"

def send_email(subject, message):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_ADDRESS
        msg.set_content(message)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

class GPUWorker:
    def __init__(self, gpu_id, puzzle_num, start_hex, end_hex, address):
        self.gpu_id = gpu_id
        self.puzzle_num = puzzle_num
        self.start = int(start_hex, 16)
        self.end = int(end_hex, 16)
        self.address = address
        self.checkpoint_file = f"checkpoint_gpu{gpu_id}_puzzle{puzzle_num}.json"
        self.keys_searched = 0
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()
        
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.keys_searched = data.get('keys_searched', 0)
                current_pos = data.get('current_position', self.start)
                print(f"[GPU {self.gpu_id}] Resuming from position: {hex(current_pos)}")
                print(f"[GPU {self.gpu_id}] Already searched: {self.keys_searched:,} keys")
                return current_pos
        return self.start
    
    def save_checkpoint(self, current_position, keys_per_sec):
        data = {
            'gpu_id': self.gpu_id,
            'puzzle_num': self.puzzle_num,
            'current_position': current_position,
            'keys_searched': self.keys_searched,
            'keys_per_sec': keys_per_sec,
            'last_update': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_stats(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            keys_per_sec = self.keys_searched / elapsed
            total_keys = self.end - self.start
            percent = (self.keys_searched / total_keys) * 100 if total_keys > 0 else 0
            remaining_keys = total_keys - self.keys_searched
            eta_seconds = remaining_keys / keys_per_sec if keys_per_sec > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            
            return {
                'keys_per_sec': keys_per_sec,
                'percent': percent,
                'eta': eta,
                'elapsed': timedelta(seconds=int(elapsed))
            }
        return None

class MultiGPUSolver:
    def __init__(self, puzzles_csv):
        self.puzzles = self.load_puzzles(puzzles_csv)
        self.workers = []
        self.solver_binary = "./ultimate_puzzle_solver"
        
    def load_puzzles(self, csv_file):
        puzzles = []
        with open(csv_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    puzzles.append({
                        'number': int(parts[0]),
                        'bits': int(parts[1]),
                        'start_hex': parts[2],
                        'end_hex': parts[3],
                        'address': parts[4]
                    })
        return puzzles
    
    def split_range_for_gpus(self, start, end, num_gpus, gpu_id):
        """Split range based on GPU capability"""
        total_range = end - start
        # GPU 0 (T4) gets 20%, GPU 1 (A10) gets 80%
        if gpu_id == 0:  # T4
            gpu_start = start
            gpu_end = start + (total_range // 5)  # 20%
        else:  # A10
            gpu_start = start + (total_range // 5)
            gpu_end = end
        return gpu_start, gpu_end
    
    def run_puzzle(self, puzzle, gpu_configs):
        """Run solver on multiple GPUs for a single puzzle"""
        print(f"\n{'='*80}")
        print(f"Puzzle #{puzzle['number']} ({puzzle['bits']} bits)")
        print(f"Address: {puzzle['address']}")
        print(f"Range: {puzzle['start_hex']} to {puzzle['end_hex']}")
        print(f"GPUs: {len(gpu_configs)}")
        print(f"{'='*80}\n")
        
        # Create workers for each GPU
        for gpu_id in gpu_configs:
            gpu_start, gpu_end = self.split_range_for_gpus(
                int(puzzle['start_hex'], 16),
                int(puzzle['end_hex'], 16),
                len(gpu_configs),
                gpu_id
            )
            
            worker = GPUWorker(
                gpu_id,
                puzzle['number'],
                hex(gpu_start),
                hex(gpu_end),
                puzzle['address']
            )
            self.workers.append(worker)
            
            # Launch GPU process
            self.launch_gpu_worker(worker)
        
        # Monitor all workers
        self.monitor_workers()
    
    def launch_gpu_worker(self, worker):
        """Launch solver process for one GPU"""
        current_pos = worker.load_checkpoint()
        
        cmd = [
            self.solver_binary,
            f"--gpu={worker.gpu_id}",
            f"--start={hex(current_pos)}",
            f"--end={hex(worker.end)}",
            f"--address={worker.address}",
            f"--checkpoint={worker.checkpoint_file}"
        ]
        
        print(f"[GPU {worker.gpu_id}] Launching: {' '.join(cmd)}")
        # TODO: Launch as subprocess and monitor
    
    def monitor_workers(self):
        """Monitor all GPU workers and display progress"""
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"\n{'='*80}")
                print(f"Ultimate Bitcoin Puzzle Solver - Live Monitor")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}\n")
                
                total_keys_per_sec = 0
                
                for worker in self.workers:
                    stats = worker.calculate_stats()
                    if stats:
                        total_keys_per_sec += stats['keys_per_sec']
                        
                        print(f"GPU {worker.gpu_id} (Puzzle {worker.puzzle_num}):")
                        print(f"  Speed: {stats['keys_per_sec']/1e6:.2f} MKey/s")
                        print(f"  Progress: {stats['percent']:.4f}%")
                        print(f"  Elapsed: {stats['elapsed']}")
                        print(f"  ETA: {stats['eta']}")
                        print(f"  Searched: {worker.keys_searched:,} keys")
                        print()
                
                print(f"{'='*80}")
                print(f"TOTAL SPEED: {total_keys_per_sec/1e6:.2f} MKey/s")
                print(f"{'='*80}\n")
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\nStopping... Saving checkpoints...")
            for worker in self.workers:
                # TODO: Signal processes to stop and save
                pass

def main():
    parser = argparse.ArgumentParser(description='Ultimate Bitcoin Puzzle Solver')
    parser.add_argument('--puzzles', default='unsolved_71_99.txt', help='CSV file with puzzles')
    parser.add_argument('--gpus', default='0,1', help='Comma-separated GPU IDs')
    parser.add_argument('--puzzle', type=int, help='Specific puzzle number to solve')
    
    args = parser.parse_args()
    
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    
    solver = MultiGPUSolver(args.puzzles)
    
    if args.puzzle:
        puzzle = next((p for p in solver.puzzles if p['number'] == args.puzzle), None)
        if puzzle:
            solver.run_puzzle(puzzle, gpu_ids)
        else:
            print(f"Puzzle {args.puzzle} not found!")
    else:
        # Run all puzzles in sequence
        for puzzle in solver.puzzles:
            solver.run_puzzle(puzzle, gpu_ids)
            # TODO: Check if found before moving to next

if __name__ == '__main__':
    main()
