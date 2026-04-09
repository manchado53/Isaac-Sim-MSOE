"""
Production-Ready IFC to USD Converter
Fully headless, no GUI required

Author: Steven Thomas
Institution: Milwaukee School of Engineering (MSOE)
Project: Robotics Research - Isaac Sim Navigation Training
Date: 2026

Purpose: 
Convert Revit buildings (via IFC intermediate format) to USD format for 
NVIDIA Isaac Sim robot training. Automatically filters MEP systems and 
other non-navigation elements.

Workflow:
    Revit (.rvt) → [Manual IFC export] → IFC (.ifc) → [This script] → USD (.usd)

Usage:
    python revit_to_usd.py input.ifc output.usd
    python revit_to_usd.py --batch input_folder/ output_folder/
"""

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import yaml

try:
    import ifcopenshell
    import ifcopenshell.geom
except ImportError:
    print("ERROR: ifcopenshell not installed")
    print("Install with: pip install ifcopenshell")
    sys.exit(1)

try:
    from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade
except ImportError:
    print("ERROR: USD Python not installed")
    print("Install with: pip install usd-core")
    sys.exit(1)


class IFCtoUSDConverter:
    """Main converter class - handles IFC parsing and USD creation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize converter with configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract config values FIRST (before using them)
        self.verbose = self.config['processing']['verbose_logging']
        self.skip_on_error = self.config['processing']['skip_on_error']
        self.exclude_types = self.config['filtering']['exclude_types']
        self.include_types = self.config['filtering']['include_types']
        
        # Setup logging (now self.verbose is available)
        self._setup_logging()
        
        self.logger.info("IFC to USD Converter initialized")
        self.logger.info(f"Config loaded from: {config_path}")
    
    
    def _setup_logging(self):
        """Setup logging with file and console output"""
        
        # Create logs directory
        log_dir = self.config['paths']['log_folder']
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('IFCtoUSD')
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # File handler
        log_file = os.path.join(
            log_dir, 
            f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    
    def process_file(self, ifc_path: str, usd_path: str) -> bool:
        """
        Process a single IFC file to USD
        
        Args:
            ifc_path: Path to input IFC file
            usd_path: Path to output USD file
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        self.logger.info("=" * 70)
        self.logger.info(f"Processing: {ifc_path}")
        self.logger.info("=" * 70)
        
        try:
            # Validate input
            if not os.path.exists(ifc_path):
                self.logger.error(f"Input file not found: {ifc_path}")
                return False
            
            # Create output directory
            os.makedirs(os.path.dirname(usd_path), exist_ok=True)
            
            # Step 1: Parse IFC
            self.logger.info("Step 1/4: Parsing IFC file...")
            ifc_data = self._parse_ifc(ifc_path)
            
            if not ifc_data:
                self.logger.error("IFC parsing failed")
                return False
            
            self.logger.info(f"✓ Parsed {ifc_data['total_elements']} elements")
            
            # Step 2: Filter elements
            self.logger.info("Step 2/4: Filtering elements...")
            filtered_data = self._filter_elements(ifc_data)
            
            self.logger.info(f"✓ Filtered to {filtered_data['total_elements']} elements")
            self._log_element_counts(filtered_data['counts'])
            
            # Step 3: Create USD
            self.logger.info("Step 3/4: Creating USD...")
            self._create_usd(filtered_data, usd_path)
            
            # Verify output
            if not os.path.exists(usd_path):
                self.logger.error("USD file was not created")
                return False
            
            file_size = os.path.getsize(usd_path) / 1024 / 1024  # MB
            self.logger.info(f"✓ Created USD: {usd_path} ({file_size:.2f} MB)")
            
            # Step 4: Post-process
            self.logger.info("Step 4/4: Post-processing...")
            self._post_process_usd(usd_path, filtered_data['counts'])
            
            self.logger.info("✓ Post-processing complete")
            
            # Generate report
            self._generate_report(ifc_path, usd_path, filtered_data)
            
            self.logger.info("=" * 70)
            self.logger.info(f"✓✓✓ SUCCESS: {usd_path}")
            self.logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return False
    
    
    def _parse_ifc(self, ifc_path: str) -> Dict[str, Any]:
        """Parse IFC file and extract geometry"""
        
        try:
            # Open IFC file
            ifc_file = ifcopenshell.open(ifc_path)
            
            # Setup geometry settings
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            
            # Extract metadata
            project = ifc_file.by_type('IfcProject')[0] if ifc_file.by_type('IfcProject') else None
            building = ifc_file.by_type('IfcBuilding')[0] if ifc_file.by_type('IfcBuilding') else None
            
            ifc_data = {
                'file': ifc_file,
                'settings': settings,
                'elements': [],
                'metadata': {
                    'project_name': project.Name if project else 'Unknown',
                    'building_name': building.Name if building else 'Unknown',
                    'ifc_schema': ifc_file.schema,
                },
                'total_elements': 0,
                'counts': {}
            }
            
            # Get all products
            products = ifc_file.by_type('IfcProduct')
            
            self.logger.info(f"Found {len(products)} total IFC products")
            
            successful = 0
            failed = 0
            
            for idx, product in enumerate(products):
                element_type = product.is_a()
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"  Processing {idx + 1}/{len(products)} products...")
                
                try:
                    # Create shape
                    shape = ifcopenshell.geom.create_shape(settings, product)
                    
                    if not shape:
                        failed += 1
                        continue
                    
                    # Extract geometry
                    geometry = shape.geometry
                    
                    # Get vertices and faces
                    verts = geometry.verts  # Flat array: [x,y,z, x,y,z, ...]
                    faces = geometry.faces  # Flat array: [i,j,k, i,j,k, ...]
                    
                    # Skip if no geometry
                    if len(verts) == 0 or len(faces) == 0:
                        failed += 1
                        continue
                    
                    # Use identity matrix (geometry already in world coords)
                    # This avoids transformation matrix extraction issues
                    matrix = (1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0)
                    
                    # Get material (optional)
                    material = None
                    try:
                        if hasattr(geometry, 'materials') and geometry.materials:
                            mat = geometry.materials[0]
                            material = {
                                'diffuse': mat.diffuse if hasattr(mat, 'diffuse') else (0.8, 0.8, 0.8),
                                'name': mat.name if hasattr(mat, 'name') else 'Material'
                            }
                    except:
                        pass  # Material is optional
                    
                    # Store element
                    element = {
                        'guid': product.GlobalId,
                        'name': product.Name or f"{element_type}_{len(ifc_data['elements'])}",
                        'type': element_type,
                        'vertices': verts,
                        'faces': faces,
                        'matrix': matrix,
                        'material': material
                    }
                    
                    ifc_data['elements'].append(element)
                    ifc_data['total_elements'] += 1
                    successful += 1
                    
                    # Count by type
                    ifc_data['counts'][element_type] = ifc_data['counts'].get(element_type, 0) + 1
                    
                except Exception as e:
                    failed += 1
                    if self.skip_on_error:
                        self.logger.debug(f"Skipped element {product.GlobalId}: {str(e)}")
                        continue
                    else:
                        raise
            
            self.logger.info(f"Geometry extraction: {successful} successful, {failed} failed")
            
            return ifc_data
            
        except Exception as e:
            self.logger.error(f"IFC parsing error: {str(e)}")
            return None
    
    
    def _filter_elements(self, ifc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter elements based on configuration"""
        
        filtered = {
            'elements': [],
            'counts': {},
            'total_elements': 0,
            'metadata': ifc_data['metadata']
        }
        
        for element in ifc_data['elements']:
            element_type = element['type']
            
            # Check exclusion list
            if element_type in self.exclude_types:
                self.logger.debug(f"Excluded: {element_type} - {element['name']}")
                continue
            
            # Check inclusion list (if specified)
            if self.include_types and element_type not in self.include_types:
                self.logger.debug(f"Not in include list: {element_type}")
                continue
            
            # Categorize
            category = self._categorize_element(element_type)
            element['category'] = category
            
            # Add to filtered data
            filtered['elements'].append(element)
            filtered['counts'][category] = filtered['counts'].get(category, 0) + 1
            filtered['total_elements'] += 1
        
        return filtered
    
    
    def _categorize_element(self, ifc_type: str) -> str:
        """Map IFC type to simplified category"""
        
        category_map = {
            # Walls
            'IfcWall': 'walls',
            'IfcWallStandardCase': 'walls',
            'IfcCurtainWall': 'curtain_walls',
            'IfcWallElementedCase': 'walls',
            
            # Curtain Wall Components (glass facades)
            'IfcMember': 'curtain_wall_mullions',
            'IfcPlate': 'curtain_wall_panels',
            
            # Doors & Windows
            'IfcDoor': 'doors',
            'IfcDoorStandardCase': 'doors',
            'IfcWindow': 'windows',
            'IfcWindowStandardCase': 'windows',
            
            # Floors, Roofs, Foundations
            'IfcSlab': 'floors',
            'IfcSlabStandardCase': 'floors',
            'IfcSlabElementedCase': 'floors',
            'IfcRoof': 'roofs',
            'IfcFooting': 'foundations',
            
            # Stairs & Circulation
            'IfcStair': 'stairs',
            'IfcStairFlight': 'stairs',
            'IfcRamp': 'ramps',
            'IfcRampFlight': 'ramps',
            'IfcRailing': 'railings',
            
            # Structure
            'IfcColumn': 'columns',
            'IfcColumnStandardCase': 'columns',
            'IfcBeam': 'beams',
            'IfcBeamStandardCase': 'beams',
            'IfcBrace': 'bracing',
            
            # Ceilings & Coverings
            'IfcCovering': 'ceilings',
            
            # Spaces & Furniture
            'IfcSpace': 'rooms',
            'IfcFurniture': 'furniture',
            'IfcFurnishingElement': 'furniture',
            'IfcSystemFurnitureElement': 'furniture',
            
            # Generic/Complex Elements
            'IfcBuildingElementProxy': 'building_elements',
            'IfcBuildingElementPart': 'building_elements',
            'IfcElementAssembly': 'building_elements',
            'IfcDiscreteAccessory': 'accessories',
            'IfcTransportElement': 'transport',
            
            # Building Structure (hierarchy elements)
            'IfcBuildingStorey': 'structure_hierarchy',
            'IfcBuilding': 'structure_hierarchy',
            'IfcSite': 'structure_hierarchy',
        }
        
        return category_map.get(ifc_type, 'other')
    
    
    def _create_usd(self, filtered_data: Dict[str, Any], usd_path: str):
        """Create USD file from filtered data"""
        
        # Create stage
        stage = Usd.Stage.CreateNew(usd_path)

        # Declare coordinate system: IFC is Z-up, set this so Isaac Sim interprets correctly
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Set units to meters (IFC default unit)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Set metadata (safe version - no 'creator' key)
        if self.config['output']['add_metadata']:
            try:
                stage.SetMetadata('comment', f"Generated from IFC on {datetime.now().isoformat()}")
            except Exception as e:
                self.logger.debug(f"Could not set metadata: {e}")
        
        # Create root prim
        # Create root prim
        root_name = self.config['output']['default_prim_name']
        root_xform = UsdGeom.Xform.Define(stage, f'/{root_name}')
        stage.SetDefaultPrim(root_xform.GetPrim())

        # Apply rotation if configured
        if self.config['output'].get('apply_rotation', False):
            rotation_axis = self.config['output'].get('rotation_axis', 'z')
            rotation_degrees = self.config['output'].get('rotation_degrees', 180)
    
            # Map axis string to vector
            axis_map = {
                'x': Gf.Vec3d(1, 0, 0),
                'y': Gf.Vec3d(0, 1, 0),
                'z': Gf.Vec3d(0, 0, 1)
            }
    
            axis_vector = axis_map.get(rotation_axis.lower(), Gf.Vec3d(0, 0, 1))
    
            # Create rotation matrix
            import math
            rotation_matrix = Gf.Matrix4d().SetRotate(
                Gf.Rotation(axis_vector, rotation_degrees)
            )
    
            # Apply transform
            root_xform.AddTransformOp().Set(rotation_matrix)
    
            self.logger.info(f"Applied {rotation_degrees}° rotation around {rotation_axis.upper()}-axis")
        
        # Create category groups
        categories = {}
        for category in set(e['category'] for e in filtered_data['elements']):
            cat_path = f'/{root_name}/{category.capitalize()}'
            cat_xform = UsdGeom.Xform.Define(stage, cat_path)
            categories[category] = cat_xform
            
            self.logger.debug(f"Created category group: {cat_path}")
        
        # Add elements
        for i, element in enumerate(filtered_data['elements']):
            try:
                self._add_element_to_stage(stage, element, categories[element['category']])
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"  Added {i + 1}/{len(filtered_data['elements'])} elements...")
                    
            except Exception as e:
                if self.skip_on_error:
                    self.logger.warning(f"Failed to add element {element['name']}: {str(e)}")
                    continue
                else:
                    raise
        
        # Save
        stage.Save()
        self.logger.debug(f"Saved USD with {len(filtered_data['elements'])} elements")
    
    
    def _add_element_to_stage(self, stage: Usd.Stage, element: Dict[str, Any], 
                              parent_xform: UsdGeom.Xform):
        """Add individual building element as USD mesh"""
        
        # Sanitize name for USD - COMPREHENSIVE VERSION
        raw_name = element['name'].strip()
        
        # Replace all non-alphanumeric characters (except underscore) with underscore
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name)
        
        # Remove multiple consecutive underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')
        
        # Ensure doesn't start with number
        if safe_name and safe_name[0].isdigit():
            safe_name = 'elem_' + safe_name
        
        # Ensure not empty
        if not safe_name or len(safe_name) < 3:
            safe_name = f"element_{element['guid'][:8]}"
        
        # Limit length and add GUID for uniqueness
        if len(safe_name) > 180:
            safe_name = safe_name[:180]
        
        safe_name = f"{safe_name}_{element['guid'][-6:]}"
        
        # Create mesh path
        mesh_path = f"{parent_xform.GetPath()}/{safe_name}"
        
        try:
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        except Exception as e:
            self.logger.warning(f"Failed to create mesh at {mesh_path}: {str(e)}")
            return  # Skip this element
        
        # Convert vertices
        verts_flat = element['vertices']
        vertices = []
        for i in range(0, len(verts_flat), 3):
            vertices.append(Gf.Vec3f(verts_flat[i], verts_flat[i+1], verts_flat[i+2]))
        
        # Convert faces
        faces_flat = element['faces']
        face_vertex_counts = []
        face_vertex_indices = []
        
        for i in range(0, len(faces_flat), 3):
            face_vertex_counts.append(3)
            face_vertex_indices.extend([faces_flat[i], faces_flat[i+1], faces_flat[i+2]])
        
        # Set mesh attributes
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        
        # Add material if available
        if self.config['output']['create_materials'] and element.get('material'):
            try:
                self._create_material(stage, mesh, element['material'])
            except Exception as e:
                self.logger.debug(f"Material creation failed: {str(e)}")
    
    
    def _create_material(self, stage: Usd.Stage, mesh: UsdGeom.Mesh, 
                         material_data: Dict[str, Any]):
        """Create and bind material to mesh"""
        
        try:
            # Create material
            mat_path = f"/Materials/{material_data['name']}"
            material = UsdShade.Material.Define(stage, mat_path)
            
            # Create shader
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            
            # Set diffuse color
            diffuse = material_data.get('diffuse', (0.8, 0.8, 0.8))
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
            
            # Connect shader to material
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # Bind material to mesh
            UsdShade.MaterialBindingAPI(mesh).Bind(material)
            
        except Exception as e:
            self.logger.debug(f"Material creation failed: {str(e)}")
    
    
    def _post_process_usd(self, usd_path: str, element_counts: Dict[str, int]):
        """Post-process USD file"""
        
        stage = Usd.Stage.Open(usd_path)
        
        # Verify default prim
        if not stage.GetDefaultPrim():
            self.logger.warning("No default prim set - attempting to fix")
            root_prims = [p for p in stage.GetPseudoRoot().GetChildren()]
            if root_prims:
                stage.SetDefaultPrim(root_prims[0])
                self.logger.info(f"Set default prim to: {root_prims[0].GetPath()}")
        
        stage.Save()
    
    
    def _log_element_counts(self, counts: Dict[str, int]):
        """Log element counts by category"""
        
        self.logger.info("  Element breakdown:")
        for category, count in sorted(counts.items()):
            self.logger.info(f"    {category.capitalize()}: {count}")
    
    
    def _generate_report(self, ifc_path: str, usd_path: str, 
                         filtered_data: Dict[str, Any]):
        """Generate JSON report of conversion"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_file': ifc_path,
            'output_file': usd_path,
            'metadata': filtered_data['metadata'],
            'element_counts': filtered_data['counts'],
            'total_elements': filtered_data['total_elements'],
            'config': self.config
        }
        
        report_path = usd_path.replace('.usd', '_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✓ Report saved: {report_path}")
    
    
    def process_batch(self, input_folder: str, output_folder: str) -> Dict[str, List[str]]:
        """Process all IFC files in a folder"""
        
        # Find all IFC files
        ifc_files = list(Path(input_folder).rglob('*.ifc'))
        
        self.logger.info(f"\nFound {len(ifc_files)} IFC files to process\n")
        
        results = {
            'success': [],
            'failed': []
        }
        
        for i, ifc_file in enumerate(ifc_files, 1):
            self.logger.info(f"\n[{i}/{len(ifc_files)}]")
            
            # Generate output path
            relative_path = ifc_file.relative_to(input_folder)
            output_usd = Path(output_folder) / relative_path.with_suffix('.usd')
            
            # Create output directory
            output_usd.parent.mkdir(parents=True, exist_ok=True)
            
            # Process
            success = self.process_file(str(ifc_file), str(output_usd))
            
            if success:
                results['success'].append(str(ifc_file))
            else:
                results['failed'].append(str(ifc_file))
        
        # Summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Success: {len(results['success'])}")
        self.logger.info(f"Failed:  {len(results['failed'])}")
        
        if results['failed']:
            self.logger.info("\nFailed files:")
            for f in results['failed']:
                self.logger.info(f"  - {f}")
        
        return results


def main():
    """Command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Convert IFC (from Revit) to USD for Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python revit_to_usd.py building.ifc building.usd
  
  # Batch processing
  python revit_to_usd.py --batch input_ifc/ output_usd/
  
  # Custom config
  python revit_to_usd.py building.ifc building.usd --config custom_config.yaml
        """
    )
    
    parser.add_argument('input', help='Input IFC file or folder')
    parser.add_argument('output', nargs='?', help='Output USD file or folder')
    parser.add_argument('--batch', action='store_true', help='Batch process folder')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch and not args.output:
        parser.error("Output path required for single file mode")
    
    # Initialize converter
    try:
        converter = IFCtoUSDConverter(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
        print("Create config.yaml or specify --config path")
        sys.exit(1)
    
    # Process
    if args.batch:
        output = args.output or 'output_usd'
        converter.process_batch(args.input, output)
    else:
        converter.process_file(args.input, args.output)


if __name__ == "__main__":
    main()