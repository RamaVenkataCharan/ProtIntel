import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { generateProteinPath } from '../utils/pathGenerator';
import { MODEL_METRICS } from '../config/modelMetrics';
import { STRUCTURE_COLORS } from '../utils/colors';

/**
 * AssemblyLoader — 3D ribbon assembly animation shown during prediction loading.
 *
 * Replaces the skeleton loader for the 3D viewer while predictions are running.
 * The ribbon assembles residue-by-residue at a pace that loosely corresponds
 * to the actual loading state (pending = slow, processing = faster, complete = instant).
 *
 * - Gracefully handles fast loads (skip animation) and slow loads (pace with job status)
 * - Falls back to a skeleton loader if WebGL init fails
 * - Respects prefers-reduced-motion
 */

interface AssemblyRibbonProps {
  revealCount: number;
  totalSegments: number;
}

const AssemblyRibbon: React.FC<AssemblyRibbonProps> = ({ revealCount, totalSegments }) => {
  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);

  const pathData = useMemo(() => {
    return generateProteinPath(
      MODEL_METRICS.demoSequence,
      MODEL_METRICS.demoQ3Prediction
    );
  }, []);

  // Pre-allocated temporaries
  const tempMatrix = useMemo(() => new THREE.Matrix4(), []);
  const tempPos = useMemo(() => new THREE.Vector3(), []);
  const tempDir = useMemo(() => new THREE.Vector3(), []);
  const tempQuat = useMemo(() => new THREE.Quaternion(), []);
  const tempScale = useMemo(() => new THREE.Vector3(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);
  const zAxis = useMemo(() => new THREE.Vector3(0, 0, 1), []);

  useFrame((state) => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    const { foldedPoints } = pathData;
    const time = state.clock.getElapsedTime();
    const limit = Math.min(totalSegments, Math.floor(revealCount));

    // Update instance count
    mesh.count = limit;
    if (limit === 0) return;

    for (let i = 0; i < limit; i++) {
      const p1 = foldedPoints[i];
      const p2 = foldedPoints[i + 1];
      if (!p1 || !p2) continue;

      tempPos.addVectors(p1, p2).multiplyScalar(0.5);
      tempDir.subVectors(p2, p1);
      const len = tempDir.length();
      if (len > 0.001) {
        tempDir.normalize();
        tempQuat.setFromUnitVectors(zAxis, tempDir);
      } else {
        tempQuat.set(0, 0, 0, 1);
      }

      const q3 = MODEL_METRICS.demoQ3Prediction[i] || 'C';
      let w = 0.3, h = 0.3;
      if (q3 === 'H') { w = 0.7; h = 0.7; }
      else if (q3 === 'E') { w = 2.0; h = 0.1; }

      // Staggered scale-in: newest segments pulse into existence
      const age = revealCount - i;
      const scaleIn = Math.min(1, age * 0.5);

      tempScale.set(w * scaleIn, h * scaleIn, (len || 0.01) * scaleIn);
      tempMatrix.compose(tempPos, tempQuat, tempScale);
      mesh.setMatrixAt(i, tempMatrix);

      // Color with subtle pulse on newest segments
      const key = (q3 === 'H' || q3 === 'E' || q3 === 'C') ? q3 : 'C';
      tempColor.set(STRUCTURE_COLORS[key].hex);

      // Newest 3 segments glow brighter
      if (age < 3) {
        const pulse = Math.sin(time * 6 + i * 0.5) * 0.15 + 1.0;
        tempColor.multiplyScalar(pulse);
      }

      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={instancedMeshRef}
      args={[null as any, null as any, totalSegments]}
      castShadow={false}
    >
      <cylinderGeometry args={[0.5, 0.5, 1, 8]} />
      <meshStandardMaterial
        roughness={0.35}
        metalness={0.15}
        emissive="#1e1b4b"
        emissiveIntensity={0.3}
      />
    </instancedMesh>
  );
};

interface AssemblyLoaderProps {
  /** Current async job status from the prediction store */
  jobStatus: 'pending' | 'processing' | 'completed' | 'failed' | null;
  /** Whether prediction has been delivered */
  isComplete: boolean;
}

export const AssemblyLoader: React.FC<AssemblyLoaderProps> = ({
  jobStatus,
  isComplete,
}) => {
  const totalSegments = MODEL_METRICS.demoSequence.length - 1;
  const [revealCount, setRevealCount] = useState(0);
  const [webglFailed, setWebglFailed] = useState(false);
  const animRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  const reducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  // WebGL availability check
  useEffect(() => {
    const timeout = setTimeout(() => {
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!gl) setWebglFailed(true);
      } catch {
        setWebglFailed(true);
      }
    }, 100);
    return () => clearTimeout(timeout);
  }, []);

  // Animation: increment revealCount at a pace driven by jobStatus
  useEffect(() => {
    if (reducedMotion || webglFailed || isComplete) return;

    const animate = (time: number) => {
      if (lastTimeRef.current === 0) lastTimeRef.current = time;
      const delta = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      let speed: number;
      if (isComplete) {
        speed = 80; // Rush to finish
      } else if (jobStatus === 'processing') {
        speed = 8; // Building anticipation
      } else {
        speed = 2; // Slow simulated waiting
      }

      setRevealCount((prev) => {
        const next = prev + delta * speed;
        if (next >= totalSegments) return totalSegments;
        return next;
      });

      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(animRef.current);
      lastTimeRef.current = 0;
    };
  }, [jobStatus, isComplete, totalSegments, reducedMotion, webglFailed]);

  // If complete, fast-forward
  useEffect(() => {
    if (isComplete) {
      setRevealCount(totalSegments);
    }
  }, [isComplete, totalSegments]);

  // Fallback: skeleton loader
  if (webglFailed || reducedMotion) {
    return <SkeletonLoader />;
  }

  return (
    <div className="w-full h-[460px] bg-[radial-gradient(circle_at_50%_30%,#161b2e_0%,#090d18_70%,#04060b_100%)] border border-[var(--border-subtle)] rounded-3xl relative overflow-hidden flex flex-col items-center justify-center">
      {/* Status label */}
      <div className="absolute top-4 left-4 z-10">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-violet-500 animate-pulse" />
          <span className="text-[10px] font-mono font-bold text-violet-400 uppercase tracking-widest">
            ASSEMBLY // {jobStatus === 'processing' ? 'COMPUTING' : 'QUEUED'}
          </span>
        </div>
        <span className="text-[9px] font-mono text-slate-500 mt-0.5 block">
          {Math.floor(revealCount)}/{totalSegments} residues
        </span>
      </div>

      {/* Progress bar */}
      <div className="absolute bottom-4 left-4 right-4 z-10">
        <div className="h-1 bg-black/40 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-100"
            style={{
              width: `${(revealCount / totalSegments) * 100}%`,
              background: 'linear-gradient(90deg, #7B2FF7, #00D9C0)',
            }}
          />
        </div>
      </div>

      <Canvas
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ fov: 40, position: [0, 8, 30], near: 0.1, far: 200 }}
      >
        <color attach="background" args={['#070a12']} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[15, 20, 15]} intensity={1.4} color="#F8FAFC" />
        <directionalLight position={[-10, 10, 15]} intensity={0.6} color="#C4B5FD" />
        <pointLight position={[0, -15, 0]} intensity={0.5} color="#00E5CC" />

        <AssemblyRibbon revealCount={revealCount} totalSegments={totalSegments} />
      </Canvas>
    </div>
  );
};

/** Simple skeleton loader fallback matching the app's design */
const STRAND_COLORS = ['#7B2FF7', '#9B59F5', '#A16AE8', '#00D9C0', '#66E8D5', '#FFB347', '#9B59F5', '#7B2FF7'];

const SkeletonLoader: React.FC = () => (
  <div className="w-full h-[460px] bg-[radial-gradient(circle_at_50%_30%,#161b2e_0%,#090d18_70%,#04060b_100%)] border border-[var(--border-subtle)] rounded-3xl flex flex-col items-center justify-center gap-4">
    <div className="flex items-end gap-[4px] p-3 rounded-2xl bg-black/40 border border-white/[0.06]">
      {STRAND_COLORS.map((hex, i) => (
        <div
          key={i}
          className="w-[5px] rounded-full"
          style={{
            height: 28,
            backgroundColor: hex,
            animation: `residue-pulse 1.1s ease-in-out ${i * 0.08}s infinite`,
            boxShadow: `0 0 8px ${hex}80`,
          }}
        />
      ))}
    </div>
    <div className="flex flex-col items-center gap-1">
      <span
        className="text-xs font-bold text-slate-300 tracking-widest uppercase"
        style={{ fontFamily: 'var(--font-heading)' }}
      >
        PREDICTING STRUCTURE
      </span>
      <span className="text-[10px] font-mono text-slate-600 tracking-wider">
        SYS_INFERENCE // ASSEMBLING_RIBBON
      </span>
    </div>
  </div>
);

export default AssemblyLoader;
