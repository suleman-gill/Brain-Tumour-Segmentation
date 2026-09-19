#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError during {description}: {e}")
        return False
    except FileNotFoundError as e:
        print(f"\nCommand not found: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Brain Tumor Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str,
                        help='Path to BraTS2020 training data directory (required for preprocessing)')
    parser.add_argument('--numpy-dir', type=str, default='numpy_data',
                        help='Path to store/read preprocessed numpy files (default: numpy_data)')
    parser.add_argument('--output-dir', type=str, default='trained_model',
                        help='Path to save trained model outputs (default: trained_model)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch size for training (default: 5)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs (default: 80)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing step (use existing numpy data)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training step (use existing model)')
    parser.add_argument('--skip-gui', action='store_true',
                        help='Skip launching the GUI after training')
    parser.add_argument('--only-gui', action='store_true',
                        help='Only run GUI (skip preprocessing and training)')
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model file (for --only-gui or --skip-training)')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show visualization plots during preprocessing')
    
    args = parser.parse_args()
    
    if args.only_gui:
        args.skip_preprocessing = True
        args.skip_training = True
    
    if not args.skip_preprocessing and not args.data_dir:
        parser.error("--data-dir is required when preprocessing is enabled")
    
    if args.skip_training and not args.model_path:
        args.model_path = os.path.join(args.output_dir, 'best_model.h5')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*60)
    print("Brain Tumor Segmentation Pipeline")
    print("="*60)
    print(f"Data Directory:      {args.data_dir or 'N/A (skipping)'}")
    print(f"Numpy Directory:     {args.numpy_dir}")
    print(f"Output Directory:    {args.output_dir}")
    print(f"Skip Preprocessing:  {args.skip_preprocessing}")
    print(f"Skip Training:       {args.skip_training}")
    print(f"Skip GUI:            {args.skip_gui}")
    if args.model_path:
        print(f"Model Path:          {args.model_path}")
    print("="*60)
    
    if not args.skip_preprocessing:
        preprocess_cmd = [
            sys.executable, os.path.join(script_dir, 'DataPreprocessing.py'),
            '--data-dir', args.data_dir,
            '--output-dir', args.numpy_dir
        ]
        if args.show_plots:
            preprocess_cmd.append('--show-plots')
        
        if not run_command(preprocess_cmd, "Data Preprocessing"):
            print("Pipeline stopped due to preprocessing error.")
            sys.exit(1)
    else:
        print("\nSkipping preprocessing step")
    
    if not args.skip_training:
        train_cmd = [
            sys.executable, os.path.join(script_dir, 'Attention_Model_Clahe1024.py'),
            '--data-dir', args.numpy_dir,
            '--output-dir', args.output_dir,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--learning-rate', str(args.learning_rate)
        ]
        
        if not run_command(train_cmd, "Model Training"):
            print("Pipeline stopped due to training error.")
            sys.exit(1)
        
        args.model_path = os.path.join(args.output_dir, 'best_model.h5')
    else:
        print("\nSkipping training step")
    
    if not args.skip_gui:
        if not args.model_path or not os.path.exists(args.model_path):
            print(f"\nWarning: Model file not found at {args.model_path}")
            alternatives = [
                os.path.join(args.output_dir, 'my_model.h5'),
                os.path.join(args.output_dir, 'my_model.keras'),
                'best_model2.h5',
                'best_model.h5'
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    args.model_path = alt
                    print(f"Using alternative model: {alt}")
                    break
            else:
                print("No trained model found. Please train a model first or specify --model-path")
                sys.exit(1)
        
        gui_cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            os.path.join(script_dir, 'app.py'),
            '--',
            '--model-path', args.model_path
        ]
        
        print(f"\n{'='*60}")
        print("Launching GUI Application")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print("\nThe GUI will open in your browser shortly...")
        print("Press Ctrl+C to stop the application.\n")
        
        try:
            subprocess.run(gui_cmd, check=True)
        except KeyboardInterrupt:
            print("\n\nGUI stopped by user.")
        except subprocess.CalledProcessError as e:
            print(f"\nError launching GUI: {e}")
            sys.exit(1)
    else:
        print("\nSkipping GUI launch")
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)


if __name__ == '__main__':
    main()
