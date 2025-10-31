#!/usr/bin/env python3
"""
CortexOS Path Configuration Manager - COMPLETE IMPLEMENTATION
Manages all file paths and directory locations for the CortexOS system.
Run this first to configure your installation paths.

FEATURES:
✅ Interactive configuration interface
✅ Configuration file persistence
✅ Path validation and creation
✅ Advanced path management
✅ Environment variable integration
✅ Backup and recovery support
✅ Cross-platform compatibility
✅ Configuration validation
✅ Path template system
✅ Automatic directory creation
"""

import os
import json
import sys
import shutil
import platform
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

class CortexOSPathManager:
    """
    Advanced Path Configuration Manager for CortexOS
    
    Provides comprehensive path management including:
    - Interactive configuration setup
    - Configuration persistence and validation
    - Path template system with variable substitution
    - Cross-platform path handling
    - Environment variable integration
    - Backup and recovery capabilities
    - Directory creation and validation
    """
    
    def __init__(self, config_file: str = "cortexos_paths.json"):
        self.config_file = config_file
        self.backup_config_file = f"{config_file}.backup"
        self.logger = self._setup_logging()
        
        # Default path templates with variable substitution
        self.path_templates = {
            # Core system paths
            "CORTEXOS_ROOT": "{base_dir}",
            "DATA_DIR": "{base_dir}/data",
            "LOGS_DIR": "{base_dir}/logs",
            "CONFIG_DIR": "{base_dir}/config",
            "TEMP_DIR": "{base_dir}/temp",
            "CACHE_DIR": "{base_dir}/cache",
            
            # NVMe and storage paths
            "NVME_DEVICE_PATH": "{nvme_device}",
            "CUBE_STORAGE_PATH": "{storage_root}/cube_storage",
            "CONTRACT_STORAGE_PATH": "{storage_root}/contracts",
            "SECTOR_MAP_PATH": "{config_dir}/sector_map.json",
            "STORAGE_INDEX_PATH": "{config_dir}/storage_index.db",
            
            # Neural data paths
            "NEURAL_DATA_DIR": "{base_dir}/neural_data",
            "MEMORY_STORAGE_PATH": "{neural_data_dir}/memory.db",
            "RESONANCE_DATA_PATH": "{neural_data_dir}/resonance.db",
            "LEXICON_DATA_PATH": "{neural_data_dir}/lexicon.db",
            "CONTEXT_DATA_PATH": "{neural_data_dir}/context.db",
            "PATTERN_DATA_PATH": "{neural_data_dir}/patterns.db",
            
            # Processing paths
            "INGESTION_QUEUE_PATH": "{temp_dir}/ingestion_queue",
            "OUTPUT_QUEUE_PATH": "{temp_dir}/output_queue",
            "PROCESSING_TEMP_PATH": "{temp_dir}/processing",
            "STREAM_BUFFER_PATH": "{temp_dir}/stream_buffers",
            "BATCH_STAGING_PATH": "{temp_dir}/batch_staging",
            
            # Backup and recovery paths
            "BACKUP_DIR": "{base_dir}/backups",
            "RECOVERY_DIR": "{base_dir}/recovery",
            "CHECKPOINT_DIR": "{base_dir}/checkpoints",
            "ARCHIVE_DIR": "{base_dir}/archives",
            
            # Component-specific paths
            "PHASE1_DATA_DIR": "{neural_data_dir}/phase1",
            "PHASE2_DATA_DIR": "{neural_data_dir}/phase2",
            "PHASE3_DATA_DIR": "{neural_data_dir}/phase3",
            "PHASE4_DATA_DIR": "{neural_data_dir}/phase4",
            "PHASE5_DATA_DIR": "{neural_data_dir}/phase5",
            "PHASE6_DATA_DIR": "{neural_data_dir}/phase6",
            
            # Monitoring and analytics paths
            "METRICS_DIR": "{base_dir}/metrics",
            "ANALYTICS_DIR": "{base_dir}/analytics",
            "REPORTS_DIR": "{base_dir}/reports",
            "ALERTS_DIR": "{base_dir}/alerts"
        }
        
        # Current resolved paths
        self.paths = {}
        
        # Configuration variables
        self.config_vars = {
            "base_dir": "",
            "nvme_device": "",
            "storage_root": "",
            "config_dir": "",
            "temp_dir": "",
            "neural_data_dir": ""
        }
        
        # Platform-specific settings
        self.platform_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "path_separator": os.sep
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for path manager"""
        logger = logging.getLogger('CortexOS.PathManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def setup_interactive(self) -> bool:
        """
        Interactive setup for CortexOS paths with advanced configuration
        
        Returns:
            bool: True if setup completed successfully
        """
        try:
            print("🧠 CortexOS Advanced Path Configuration Setup")
            print("=" * 60)
            print("This will configure all paths for your CortexOS installation.")
            print("Press Enter to use default values shown in brackets.\n")
            
            # Get base installation directory
            default_base = str(Path.home() / "CortexOS")
            base_dir = self._get_user_input(
                "Enter CortexOS installation directory",
                default_base,
                self._validate_base_directory
            )
            
            self.config_vars["base_dir"] = base_dir
            self.config_vars["config_dir"] = str(Path(base_dir) / "config")
            self.config_vars["temp_dir"] = str(Path(base_dir) / "temp")
            self.config_vars["neural_data_dir"] = str(Path(base_dir) / "neural_data")
            
            # Storage configuration
            print("\n🔧 Storage Configuration")
            storage_type = self._get_storage_configuration()
            
            # Advanced configuration options
            print("\n⚙️ Advanced Configuration")
            self._configure_advanced_options()
            
            # Resolve all paths from templates
            self._resolve_all_paths()
            
            # Validate configuration
            if not self._validate_configuration():
                print("❌ Configuration validation failed!")
                return False
            
            # Create directories
            self._create_all_directories()
            
            # Save configuration with backup
            self._save_configuration_with_backup()
            
            # Generate environment file
            self._generate_environment_file()
            
            # Display summary
            self._display_configuration_summary()
            
            self.logger.info("CortexOS path configuration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            print(f"❌ Setup failed: {e}")
            return False
    
    def _get_user_input(self, prompt: str, default: str, validator=None) -> str:
        """Get validated user input with default value"""
        while True:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                user_input = default
            
            if validator:
                is_valid, message = validator(user_input)
                if not is_valid:
                    print(f"❌ {message}")
                    continue
            
            return user_input
    
    def _validate_base_directory(self, path: str) -> tuple[bool, str]:
        """Validate base directory path"""
        try:
            path_obj = Path(path)
            
            # Check if path is absolute
            if not path_obj.is_absolute():
                return False, "Path must be absolute"
            
            # Check if we can create the directory
            path_obj.mkdir(parents=True, exist_ok=True)
            
            # Check write permissions
            test_file = path_obj / ".cortexos_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except PermissionError:
                return False, "No write permission for this directory"
            
            return True, "Valid directory"
            
        except Exception as e:
            return False, f"Invalid path: {e}"
    
    def _get_storage_configuration(self) -> str:
        """Configure storage options"""
        print("Select storage configuration:")
        print("1. NVMe device (hardware)")
        print("2. File-based simulation")
        print("3. Hybrid (NVMe + file backup)")
        
        choice = self._get_user_input("Storage type", "2", self._validate_storage_choice)
        
        if choice == "1":
            # NVMe configuration
            nvme_path = self._get_user_input(
                "Enter NVMe device path (e.g., /dev/nvme0n1)",
                "/dev/nvme0n1",
                self._validate_nvme_device
            )
            self.config_vars["nvme_device"] = nvme_path
            self.config_vars["storage_root"] = nvme_path
            return "nvme"
            
        elif choice == "3":
            # Hybrid configuration
            nvme_path = self._get_user_input(
                "Enter NVMe device path",
                "/dev/nvme0n1",
                self._validate_nvme_device
            )
            self.config_vars["nvme_device"] = nvme_path
            self.config_vars["storage_root"] = str(Path(self.config_vars["base_dir"]) / "storage")
            return "hybrid"
            
        else:
            # File-based simulation
            self.config_vars["nvme_device"] = ""
            self.config_vars["storage_root"] = str(Path(self.config_vars["base_dir"]) / "storage")
            return "simulation"
    
    def _validate_storage_choice(self, choice: str) -> tuple[bool, str]:
        """Validate storage configuration choice"""
        if choice in ["1", "2", "3"]:
            return True, "Valid choice"
        return False, "Please enter 1, 2, or 3"
    
    def _validate_nvme_device(self, device_path: str) -> tuple[bool, str]:
        """Validate NVMe device path"""
        if not device_path:
            return True, "Empty path (simulation mode)"
        
        path_obj = Path(device_path)
        if not path_obj.exists():
            return False, f"Device {device_path} does not exist"
        
        if not os.access(device_path, os.R_OK | os.W_OK):
            return False, f"No read/write access to {device_path}"
        
        return True, "Valid NVMe device"
    
    def _configure_advanced_options(self):
        """Configure advanced options"""
        # Cache size configuration
        cache_size = self._get_user_input(
            "Cache size in MB",
            "1024",
            lambda x: (x.isdigit() and int(x) > 0, "Must be a positive number")
        )
        
        # Backup retention
        backup_retention = self._get_user_input(
            "Backup retention days",
            "30",
            lambda x: (x.isdigit() and int(x) > 0, "Must be a positive number")
        )
        
        # Log level
        log_level = self._get_user_input(
            "Log level (DEBUG/INFO/WARNING/ERROR)",
            "INFO",
            lambda x: (x.upper() in ["DEBUG", "INFO", "WARNING", "ERROR"], "Invalid log level")
        )
        
        # Store advanced configuration
        self.config_vars.update({
            "cache_size_mb": int(cache_size),
            "backup_retention_days": int(backup_retention),
            "log_level": log_level.upper()
        })
    
    def _resolve_all_paths(self):
        """Resolve all path templates with current configuration variables"""
        self.paths = {}
        
        for path_name, template in self.path_templates.items():
            try:
                # Substitute variables in template
                resolved_path = template.format(**self.config_vars)
                
                # Convert to absolute path
                if not Path(resolved_path).is_absolute():
                    resolved_path = str(Path(self.config_vars["base_dir"]) / resolved_path)
                
                # Normalize path for current platform
                self.paths[path_name] = str(Path(resolved_path).resolve())
                
            except KeyError as e:
                self.logger.warning(f"Missing variable {e} for path {path_name}")
                self.paths[path_name] = template
    
    def _validate_configuration(self) -> bool:
        """Validate the complete configuration"""
        try:
            # Check required paths are set
            required_paths = ["CORTEXOS_ROOT", "DATA_DIR", "LOGS_DIR", "CONFIG_DIR"]
            for path_name in required_paths:
                if not self.paths.get(path_name):
                    self.logger.error(f"Required path {path_name} not configured")
                    return False
            
            # Check base directory exists and is writable
            base_dir = Path(self.paths["CORTEXOS_ROOT"])
            if not base_dir.exists():
                base_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = base_dir / ".cortexos_config_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                self.logger.error(f"Cannot write to base directory: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _create_all_directories(self):
        """Create all required directories"""
        directories_created = 0
        directories_failed = 0
        
        for path_name, path_value in self.paths.items():
            if path_name.endswith("_DIR") or path_name.endswith("_PATH"):
                try:
                    # Skip file paths, only create directories
                    if path_name.endswith("_PATH") and "." in Path(path_value).name:
                        # This is a file path, create parent directory
                        Path(path_value).parent.mkdir(parents=True, exist_ok=True)
                    else:
                        # This is a directory path
                        Path(path_value).mkdir(parents=True, exist_ok=True)
                    
                    directories_created += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create directory {path_value}: {e}")
                    directories_failed += 1
        
        print(f"📁 Created {directories_created} directories")
        if directories_failed > 0:
            print(f"⚠️ Failed to create {directories_failed} directories")
    
    def _save_configuration_with_backup(self):
        """Save configuration with backup"""
        try:
            # Create backup of existing config
            if os.path.exists(self.config_file):
                shutil.copy2(self.config_file, self.backup_config_file)
            
            # Prepare configuration data
            config_data = {
                "metadata": {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "platform": self.platform_info,
                    "config_vars": self.config_vars
                },
                "paths": self.paths,
                "templates": self.path_templates
            }
            
            # Save configuration
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4, sort_keys=True)
            
            print(f"💾 Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _generate_environment_file(self):
        """Generate environment file for easy path access"""
        env_file = Path(self.config_vars["base_dir"]) / "cortexos_env.sh"
        
        try:
            with open(env_file, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# CortexOS Environment Variables\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                
                for path_name, path_value in self.paths.items():
                    f.write(f'export {path_name}="{path_value}"\n')
                
                f.write("\necho 'CortexOS environment variables loaded'\n")
            
            # Make executable
            env_file.chmod(0o755)
            print(f"🔧 Environment file created: {env_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create environment file: {e}")
    
    def _display_configuration_summary(self):
        """Display configuration summary"""
        print("\n" + "=" * 60)
        print("🎉 CortexOS Configuration Summary")
        print("=" * 60)
        print(f"📁 Installation Directory: {self.paths['CORTEXOS_ROOT']}")
        print(f"💾 Storage Type: {self.config_vars.get('storage_type', 'simulation')}")
        print(f"🔧 Configuration File: {self.config_file}")
        print(f"📊 Total Paths Configured: {len(self.paths)}")
        print(f"🖥️ Platform: {self.platform_info['system']} {self.platform_info['machine']}")
        
        print("\n📋 Key Directories:")
        key_dirs = ["DATA_DIR", "LOGS_DIR", "CONFIG_DIR", "NEURAL_DATA_DIR", "BACKUP_DIR"]
        for dir_name in key_dirs:
            if dir_name in self.paths:
                print(f"  {dir_name}: {self.paths[dir_name]}")
        
        print("\n✅ Configuration completed successfully!")
        print("You can now run the CortexOS system components.")
    
    def load_configuration(self) -> bool:
        """
        Load configuration from JSON file with validation
        
        Returns:
            bool: True if configuration loaded successfully
        """
        try:
            if not os.path.exists(self.config_file):
                self.logger.info("No existing configuration found")
                return False
            
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load paths
            self.paths = config_data.get("paths", {})
            
            # Load configuration variables
            metadata = config_data.get("metadata", {})
            self.config_vars = metadata.get("config_vars", {})
            
            # Load templates if available
            if "templates" in config_data:
                self.path_templates = config_data["templates"]
            
            # Validate loaded configuration
            if not self._validate_loaded_configuration():
                self.logger.warning("Loaded configuration failed validation")
                return False
            
            self.logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _validate_loaded_configuration(self) -> bool:
        """Validate loaded configuration"""
        try:
            # Check if required paths exist
            required_paths = ["CORTEXOS_ROOT", "DATA_DIR", "LOGS_DIR"]
            for path_name in required_paths:
                if path_name not in self.paths:
                    return False
                
                # Check if path exists
                if not Path(self.paths[path_name]).exists():
                    self.logger.warning(f"Path {path_name} does not exist: {self.paths[path_name]}")
            
            return True
            
        except Exception:
            return False
    
    def get_path(self, path_name: str) -> str:
        """
        Get a specific path by name
        
        Args:
            path_name: Name of the path to retrieve
            
        Returns:
            str: The path value or empty string if not found
        """
        return self.paths.get(path_name, "")
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all configured paths"""
        return self.paths.copy()
    
    def update_path(self, path_name: str, path_value: str) -> bool:
        """
        Update a specific path
        
        Args:
            path_name: Name of the path to update
            path_value: New path value
            
        Returns:
            bool: True if update successful
        """
        try:
            # Validate path
            path_obj = Path(path_value)
            if not path_obj.is_absolute():
                path_value = str(Path(self.config_vars["base_dir"]) / path_value)
            
            self.paths[path_name] = str(Path(path_value).resolve())
            
            # Save updated configuration
            self._save_configuration_with_backup()
            
            self.logger.info(f"Updated path {path_name} to {path_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update path {path_name}: {e}")
            return False
    
    def create_path(self, path_name: str) -> bool:
        """
        Create directory for a specific path
        
        Args:
            path_name: Name of the path to create
            
        Returns:
            bool: True if creation successful
        """
        try:
            path_value = self.get_path(path_name)
            if not path_value:
                return False
            
            # Create directory
            Path(path_value).mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Created directory for {path_name}: {path_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create path {path_name}: {e}")
            return False
    
    def validate_path(self, path_name: str) -> tuple[bool, str]:
        """
        Validate a specific path
        
        Args:
            path_name: Name of the path to validate
            
        Returns:
            tuple: (is_valid, message)
        """
        try:
            path_value = self.get_path(path_name)
            if not path_value:
                return False, f"Path {path_name} not configured"
            
            path_obj = Path(path_value)
            
            # Check if path exists
            if not path_obj.exists():
                return False, f"Path does not exist: {path_value}"
            
            # Check permissions
            if not os.access(path_value, os.R_OK):
                return False, f"No read access: {path_value}"
            
            return True, "Path is valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def backup_configuration(self) -> bool:
        """Create backup of current configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.config_file}.backup_{timestamp}"
            
            shutil.copy2(self.config_file, backup_file)
            
            self.logger.info(f"Configuration backed up to {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup configuration: {e}")
            return False
    
    def restore_configuration(self, backup_file: str = None) -> bool:
        """Restore configuration from backup"""
        try:
            if not backup_file:
                backup_file = self.backup_config_file
            
            if not os.path.exists(backup_file):
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            shutil.copy2(backup_file, self.config_file)
            
            # Reload configuration
            self.load_configuration()
            
            self.logger.info(f"Configuration restored from {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration: {e}")
            return False
    
    def export_paths_to_env(self, env_file: str = None) -> bool:
        """Export paths as environment variables"""
        try:
            if not env_file:
                env_file = "cortexos_paths.env"
            
            with open(env_file, 'w') as f:
                f.write("# CortexOS Path Environment Variables\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                
                for path_name, path_value in self.paths.items():
                    f.write(f'{path_name}="{path_value}"\n')
            
            self.logger.info(f"Paths exported to {env_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export paths: {e}")
            return False
    
    def update_file_with_paths(self, file_path: str) -> bool:
        """
        Update a Python file with the configured paths using placeholder replacement
        
        Args:
            file_path: Path to the file to update
            
        Returns:
            bool: True if update successful
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace path placeholders
            updated = False
            for path_name, path_value in self.paths.items():
                placeholder = f"{{PATH_{path_name}}}"
                if placeholder in content:
                    content = content.replace(placeholder, path_value)
                    updated = True
            
            # Write updated content if changes were made
            if updated:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.logger.info(f"Updated file with paths: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update file {file_path}: {e}")
            return False
    
    def update_all_files(self, directory: str) -> List[str]:
        """
        Update all Python files in a directory with configured paths
        
        Args:
            directory: Directory to scan for Python files
            
        Returns:
            List[str]: List of updated file paths
        """
        updated_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        if self.update_file_with_paths(file_path):
                            updated_files.append(file_path)
            
            self.logger.info(f"Updated {len(updated_files)} files in {directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to update files in {directory}: {e}")
        
        return updated_files
    
    def get_configuration_status(self) -> Dict:
        """Get comprehensive configuration status"""
        status = {
            "configured": bool(self.paths),
            "config_file_exists": os.path.exists(self.config_file),
            "total_paths": len(self.paths),
            "valid_paths": 0,
            "invalid_paths": 0,
            "missing_directories": [],
            "platform": self.platform_info
        }
        
        # Check path validity
        for path_name, path_value in self.paths.items():
            is_valid, _ = self.validate_path(path_name)
            if is_valid:
                status["valid_paths"] += 1
            else:
                status["invalid_paths"] += 1
                if not Path(path_value).exists():
                    status["missing_directories"].append(path_name)
        
        return status
    
    def print_configuration_status(self):
        """Print detailed configuration status"""
        status = self.get_configuration_status()
        
        print("\n🔍 CortexOS Path Configuration Status")
        print("=" * 50)
        print(f"Configured: {'✅' if status['configured'] else '❌'}")
        print(f"Config File: {'✅' if status['config_file_exists'] else '❌'} ({self.config_file})")
        print(f"Total Paths: {status['total_paths']}")
        print(f"Valid Paths: {status['valid_paths']}")
        print(f"Invalid Paths: {status['invalid_paths']}")
        
        if status['missing_directories']:
            print(f"\n❌ Missing Directories ({len(status['missing_directories'])}):")
            for path_name in status['missing_directories']:
                print(f"  - {path_name}: {self.get_path(path_name)}")
        
        print(f"\n🖥️ Platform: {status['platform']['system']} {status['platform']['machine']}")

def main():
    """Main entry point for path manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CortexOS Path Configuration Manager')
    parser.add_argument('--config', default='cortexos_paths.json', 
                       help='Configuration file path')
    parser.add_argument('--setup', action='store_true', 
                       help='Run interactive setup')
    parser.add_argument('--status', action='store_true', 
                       help='Show configuration status')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate current configuration')
    parser.add_argument('--export-env', metavar='FILE', 
                       help='Export paths to environment file')
    parser.add_argument('--update-files', metavar='DIR', 
                       help='Update Python files in directory with paths')
    
    args = parser.parse_args()
    
    # Create path manager
    manager = CortexOSPathManager(args.config)
    
    # Handle commands
    if args.setup:
        # Run interactive setup
        if manager.load_configuration():
            print("📋 Existing configuration found.")
            choice = input("Do you want to reconfigure? (y/n): ").strip().lower()
            if choice != 'y':
                print("Setup cancelled.")
                return
        
        if manager.setup_interactive():
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    
    elif args.status:
        # Show status
        if manager.load_configuration():
            manager.print_configuration_status()
        else:
            print("❌ No configuration found. Run with --setup to configure.")
    
    elif args.validate:
        # Validate configuration
        if manager.load_configuration():
            status = manager.get_configuration_status()
            if status['invalid_paths'] == 0:
                print("✅ All paths are valid!")
            else:
                print(f"❌ {status['invalid_paths']} invalid paths found.")
                manager.print_configuration_status()
        else:
            print("❌ No configuration found.")
    
    elif args.export_env:
        # Export to environment file
        if manager.load_configuration():
            if manager.export_paths_to_env(args.export_env):
                print(f"✅ Paths exported to {args.export_env}")
            else:
                print("❌ Export failed!")
        else:
            print("❌ No configuration found.")
    
    elif args.update_files:
        # Update files in directory
        if manager.load_configuration():
            updated_files = manager.update_all_files(args.update_files)
            print(f"✅ Updated {len(updated_files)} files in {args.update_files}")
        else:
            print("❌ No configuration found.")
    
    else:
        # Default behavior - load existing or run setup
        if manager.load_configuration():
            print("📋 Existing configuration loaded.")
            manager.print_configuration_status()
            
            choice = input("\nDo you want to reconfigure? (y/n): ").strip().lower()
            if choice == 'y':
                manager.setup_interactive()
        else:
            print("🔧 No configuration found. Running interactive setup...")
            manager.setup_interactive()

if __name__ == "__main__":
    main()

