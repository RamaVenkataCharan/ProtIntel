import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { ProteinRibbon } from './ProteinRibbon';
import { RotateCcw, Compass, HelpCircle } from 'lucide-react';
import { STRUCTURE_COLORS, NEUTRAL_COLORS } from '../utils/colors';

interface ProteinStructure3DProps {
  sequence: string | null;
  q3Prediction: string[] | null;
  q8Prediction?: string[] | null;
  confidence: number[] | null;
  hoveredIndex: number | null;
  isPredicting: boolean;
}

// CameraRig handles the dollying entrance transition
const CameraRig: React.FC<{ reducedMotion: boolean }> = ({ reducedMotion }) => {
  const isInitial = useRef(true);

  useFrame((state) => {
    if (isInitial.current) {
      if (reducedMotion) {
        state.camera.position.set(0, 0, 25);
        state.camera.lookAt(0, 0, 0);
        isInitial.current = false;
      } else {
        state.camera.position.set(0, 10, 50);
        isInitial.current = false;
      }
    }

    if (!reducedMotion && state.clock.getElapsedTime() < 1.8) {
      const t = state.clock.getElapsedTime() / 1.8;
      const ease = 1 - Math.pow(1 - t, 3); // Cubic ease out
      state.camera.position.set(
        0,
        THREE.MathUtils.lerp(10, 0, ease),
        THREE.MathUtils.lerp(50, 25, ease)
      );
      state.camera.lookAt(0, 0, 0);
    }
  });

  return null;
};

// GroupRig handles slow idle rotation (pauses during drag/orbit interactions)
interface GroupRigProps {
  children: React.ReactNode;
  reducedMotion: boolean;
  isInteracting: React.MutableRefObject<boolean>;
}

const GroupRig: React.FC<GroupRigProps> = ({ children, reducedMotion, isInteracting }) => {
  const groupRef = useRef<THREE.Group>(null);
  const currentRotation = useRef(0);

  useFrame((_, delta) => {
    if (groupRef.current && !isInteracting.current && !reducedMotion) {
      currentRotation.current += delta * 0.05; // 0.05 rad/sec (slow ambient spin)
      groupRef.current.rotation.y = currentRotation.current;
    }
  });

  return <group ref={groupRef}>{children}</group>;
};

// Running wave component used as a waiting/loading visual during inference
const LoadingRibbon: React.FC = () => {
  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  const L = 60; // Mock sequence length
  const tempMatrix = new THREE.Matrix4();
  const tempPosition = new THREE.Vector3();
  const tempRotation = new THREE.Quaternion();
  const tempScale = new THREE.Vector3();
  const tempColor = new THREE.Color(NEUTRAL_COLORS.loadingHex); // Loading/Neutral Indigo color
  const zAxis = new THREE.Vector3(0, 0, 1);

  useFrame((state) => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    const time = state.clock.getElapsedTime();
    const spacing = 0.8;
    const startX = -((L - 1) * spacing) / 2;

    for (let i = 0; i < L - 1; i++) {
      const x1 = startX + i * spacing;
      const x2 = startX + (i + 1) * spacing;

      // Vertical sine wave ripple animation
      const y1 = Math.sin(time * 5 + i * 0.25) * 0.8;
      const y2 = Math.sin(time * 5 + (i + 1) * 0.25) * 0.8;

      const p1 = new THREE.Vector3(x1, y1, 0);
      const p2 = new THREE.Vector3(x2, y2, 0);

      tempPosition.addVectors(p1, p2).multiplyScalar(0.5);
      const dir = new THREE.Vector3().subVectors(p2, p1);
      const len = dir.length();
      dir.normalize();

      tempRotation.setFromUnitVectors(zAxis, dir);
      tempScale.set(0.35, 0.35, len);

      tempMatrix.compose(tempPosition, tempRotation, tempScale);
      mesh.setMatrixAt(i, tempMatrix);
      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={instancedMeshRef} args={[null as any, null as any, L - 1]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.4} metalness={0.2} />
    </instancedMesh>
  );
};

export const ProteinStructure3D: React.FC<ProteinStructure3DProps> = ({
  sequence,
  q3Prediction,
  q8Prediction,
  confidence,
  hoveredIndex,
  isPredicting,
}) => {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [resetTrigger, setResetTrigger] = useState(0);
  const isInteracting = useRef(false);

  // Detect reduced-motion preferences
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mediaQuery.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Detect mobile/resize for geometry scaling
  useEffect(() => {
    const checkDevice = () => {
      setIsMobile(window.innerWidth < 768 || navigator.maxTouchPoints > 1);
    };
    checkDevice();
    window.addEventListener('resize', checkDevice);
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  const triggerReset = () => {
    setResetTrigger((prev) => prev + 1);
  };

  const hasData = sequence && q3Prediction && confidence;

  return (
    <div className="w-full h-[400px] bg-[radial-gradient(circle_at_30%_20%,#1a0b2e_0%,#0a0e17_60%,#050508_100%)] border border-purple-900/30 rounded-3xl relative overflow-hidden flex flex-col group/canvas">
      {/* Controls Overlay */}
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        <button
          onClick={triggerReset}
          disabled={!hasData || isPredicting}
          className="bg-slate-900/90 border border-slate-800 text-slate-300 hover:text-white p-2.5 rounded-xl hover:bg-slate-800 transition-all flex items-center gap-1.5 text-xs font-semibold cursor-pointer disabled:opacity-30 disabled:pointer-events-none"
          title="Play fold-in reveal again"
        >
          <RotateCcw className="h-4 w-4" />
          <span>Fold Reveal</span>
        </button>

        <div className="bg-slate-900/90 border border-slate-800 text-slate-400 p-2.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 select-none">
          <Compass className="h-4 w-4 text-purple-400" />
          <span>Drag to Orbit / Scroll to Zoom</span>
        </div>
      </div>

      {/* Dynamic Status / Indicators */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        {reducedMotion && (
          <span className="bg-slate-900/90 border border-slate-800 text-[10px] text-amber-400 font-bold px-2 py-1 rounded-lg select-none">
            Reduced Motion Active
          </span>
        )}
        {isMobile && (
          <span className="bg-slate-900/90 border border-slate-800 text-[10px] text-sky-400 font-bold px-2 py-1 rounded-lg select-none">
            Mobile Mode (Low LOD)
          </span>
        )}
      </div>

      {/* R3F Canvas */}
      <div className="w-full flex-1">
        <Canvas
          shadows
          gl={{ antialias: !isMobile, powerPreference: "high-performance" }}
          camera={{ fov: 40, near: 0.1, far: 100, position: [0, 0, 25] }}
        >
          <color attach="background" args={['#0a0e17']} />
          <fog attach="fog" args={['#0a0e17', 15, 40]} />

          {/* Lighting Rig */}
          <ambientLight intensity={0.6} />
          <directionalLight
            position={[10, 20, 10]}
            intensity={1.2}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <pointLight position={[-10, -10, -10]} intensity={0.4} color="#00D9C0" />
          <pointLight position={[0, 0, 15]} intensity={0.5} color="#7B2FF7" />

          {/* Camera Dolly Trigger */}
          <CameraRig reducedMotion={reducedMotion} />

          {/* Orbit Controls (pauses idle spin on start) */}
          <OrbitControls
            enableDamping
            dampingFactor={0.05}
            maxDistance={80}
            minDistance={10}
            onStart={() => {
              isInteracting.current = true;
            }}
            onEnd={() => {
              isInteracting.current = false;
            }}
          />

          {/* Idle Rotation Group Wrapper */}
          <GroupRig reducedMotion={reducedMotion} isInteracting={isInteracting}>
            {isPredicting ? (
              <LoadingRibbon />
            ) : hasData ? (
              <ProteinRibbon
                sequence={sequence}
                q3Prediction={q3Prediction}
                q8Prediction={q8Prediction}
                confidence={confidence}
                hoveredIndex={hoveredIndex}
                resetTrigger={resetTrigger}
                reducedMotion={reducedMotion}
                isMobile={isMobile}
              />
            ) : (
              // Empty/Neutral structure placeholder
              <mesh>
                <boxGeometry args={[1, 1, 1]} />
                <meshStandardMaterial roughness={0.8} color="#1e293b" />
              </mesh>
            )}
          </GroupRig>
        </Canvas>
      </div>

      {/* Floating Explanatory Footer */}
      {!hasData && !isPredicting && (
        <div className="absolute inset-0 bg-slate-950/20 pointer-events-none flex flex-col items-center justify-center p-6 text-center">
          <HelpCircle className="h-10 w-10 text-slate-800 mb-2 animate-pulse" />
          <p className="text-xs font-semibold text-slate-500 max-w-xs">
            Run a prediction to render the 3D secondary structure folding sequence.
          </p>
        </div>
      )}

      {isPredicting && (
        <div className="absolute bottom-4 left-4 right-4 z-10 bg-slate-900/90 border border-slate-800 p-3 rounded-2xl flex items-center justify-center gap-3 backdrop-blur-md">
          <div className="h-4 w-4 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin shrink-0"></div>
          <span className="text-xs font-semibold text-slate-300">
            ESM-2 Predicting Folding Backbone...
          </span>
        </div>
      )}

      {hasData && !isPredicting && (
        <div className="bg-slate-950 border-t border-slate-900 px-4 py-2 flex items-center justify-between text-[10px] text-slate-500 font-semibold select-none">
          <div className="flex gap-4">
            {Object.entries(STRUCTURE_COLORS).map(([key, config]) => (
              <span key={key} className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: config.hex }} />
                <span>{config.label}</span>
              </span>
            ))}
          </div>

          <span className="font-mono text-purple-400">
            N-Terminus &rarr; C-Terminus
          </span>
        </div>
      )}
    </div>
  );
};
export default ProteinStructure3D;
