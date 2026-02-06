import os
import json
import traceback

LOG_FILE = 'sunset_log.txt'
MODEL_DIR = r'H:\from_D\SO8T_models\Qwen2.5-7B-Instruct'
REPORT_FILE = 'diagnosis_report.txt'

def diagnose():
    with open(REPORT_FILE, 'w', encoding='utf-8') as report:
        report.write("=== LOG TAIL (Finding Traceback) ===\n")
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # Find last occurrence of "Traceback"
                    traceback_start = -1
                    for i, line in enumerate(lines):
                        if "Traceback" in line:
                            traceback_start = i
                    
                    if traceback_start != -1:
                        report.write("Traceback found:\n")
                        report.write("".join(lines[traceback_start:]))
                    else:
                        report.write("No 'Traceback' string found in log. Last 20 lines:\n")
                        report.write("".join(lines[-20:]))
            else:
                report.write(f"Log file not found: {LOG_FILE}\n")
        except Exception as e:
            report.write(f"Error reading log: {e}\n")
        
        report.write("\n=== MODEL DIRECTORY CHECK ===\n")
        try:
            if os.path.exists(MODEL_DIR):
                files = os.listdir(MODEL_DIR)
                report.write(f"Contents of {MODEL_DIR}:\n")
                for f in files:
                    size = os.path.getsize(os.path.join(MODEL_DIR, f))
                    report.write(f" - {f} ({size} bytes)\n")
                
                config_path = os.path.join(MODEL_DIR, 'config.json')
                if 'config.json' in files:
                    report.write("\n=== CONFIG.JSON CHECK ===\n")
                    try:
                        with open(config_path, 'r', encoding='utf-8') as cf:
                            config_data = json.load(cf)
                            report.write("Valid JSON loaded.\n")
                            report.write(f"Keys: {list(config_data.keys())}\n")
                            if "model_type" in config_data:
                                report.write(f"Model Type: {config_data['model_type']}\n")
                    except Exception as e:
                         report.write(f"Error reading/parsing config.json: {e}\n")
                else:
                    report.write("\nCRITICAL: config.json NOT FOUND in model directory.\n")

            else:
                report.write(f"Model directory does not exist: {MODEL_DIR}\n")
        except Exception as e:
            report.write(f"Error checking model dir: {e}\n")

if __name__ == "__main__":
    diagnose()
    print("Diagnosis complete. Check diagnosis_report.txt")
