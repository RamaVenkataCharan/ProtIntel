import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { generateProteinPath } from '../utils/pathGenerator';
import { MODEL_METRICS } from '../config/modelMetrics';
import { STRUCTURE_COLORS } from '../utils/colors';

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

      const age = revealCount - i;
      const scaleIn = Math.min(1, age * 0.5);

      tempScale.set(w * scaleIn, h * scaleIn, (len || 0.01) * scaleIn);
      tempMatrix.compose(tempPos, tempQuat, tempScale);
      mesh.setMatrixAt(i, tempMatrix);

      const key = (q3 === 'H' || q3 === 'E' || q3 === 'C') ? q3 : 'C';
      tempColor.set(STRUCTURE_COLORS[key].hex);

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
        roughness={0.3}
        metalness={0.2}
        emissive="#1e1b4b"
        emissiveIntensity={0.35}
      />
    </instancedMesh>
  );
};

interface AssemblyLoaderProps {
  jobStatus: 'pending' | 'processing' | 'completed' | 'failed' | null;
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

  useEffect(() => {
    if (reducedMotion || webglFailed || isComplete) return;

    const animate = (time: number) => {
      if (lastTimeRef.current === 0) lastTimeRef.current = time;
      const delta = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      let speed: number;
      if (isComplete) {
        speed = 80;
      } else if (jobStatus === 'processing') {
        speed = 9;
      } else {
        speed = 2.5;
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

  useEffect(() => {
    if (isComplete) {
      setRevealCount(totalSegments);
    }
  }, [isComplete, totalSegments]);

  if (webglFailed || reducedMotion) {
    return <SkeletonLoader />;
  }

  const pct = Math.min(100, Math.round((revealCount / totalSegments) * 100));

  return (
    <div className="w-full h-[460px] surface-tier-2 border border-[var(--border-muted)] rounded-3xl relative overflow-hidden flex flex-col items-center justify-center">
      {/* Laser scan highlight */}
      <div className="animate-laser-scan" />

      {/* Status HUD Header */}
      <div className="absolute top-5 left-5 right-5 z-10 flex items-center justify-between">
        <div className="flex items-center gap-2.5 bg-black/50 px-3 py-1.5 rounded-xl border border-white/10 backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-[var(--aurora-teal)] animate-pulse" />
          <span className="text-[10px] font-mono font-bold text-teal-300 uppercase tracking-widest">
            RIBBON ASSEMBLY // {jobStatus === 'processing' ? 'COMPUTING_EMBEDDINGS' : 'INITIALIZING'}
          </span>
        </div>
        <span className="text-[11px] font-mono font-bold text-slate-300 bg-black/50 px-2.5 py-1 rounded-xl border border-white/10">
          {pct}% ({Math.floor(revealCount)}/{totalSegments} aa)
        </span>
      </div>

      {/* Assembly Progress Bar */}
      <div className="absolute bottom-5 left-5 right-5 z-10">
        <div className="h-1.5 bg-black/60 rounded-full overflow-hidden border border-white/10 p-[1px]">
          <div
            className="h-full rounded-full transition-all duration-100"
            style={{
              width: `${(revealCount / totalSegments) * 100}%`,
              background: 'linear-gradient(90deg, var(--aurora-violet), var(--aurora-teal))',
              boxShadow: '0 0 10px var(--aurora-teal-glow)',
            }}
          />
        </div>
      </div>

      <Canvas
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ fov: 38, position: [0, 8, 32], near: 0.1, far: 200 }}
      >
        <color attach="background" args={['#040711']} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[15, 20, 15]} intensity={1.4} color="#F8FAFC" />
        <directionalLight position={[-10, 10, 15]} intensity={0.6} color="#C4B5FD" />
        <pointLight position={[0, -15, 0]} intensity={0.5} color="#00E5CC" />

        <AssemblyRibbon revealCount={revealCount} totalSegments={totalSegments} />
      </Canvas>
    </div>
  );
};

const STRAND_COLORS = ['#7B2FF7', '#9B59F5', '#A16AE8', '#00D9C0', '#66E8D5', '#FFB347', '#9B59F5', '#7B2FF7'];

const SkeletonLoader: React.FC = () => (
  <div className="w-full h-[460px] surface-tier-2 border border-[var(--border-muted)] rounded-3xl flex flex-col items-center justify-center gap-4">
    <div className="flex items-end gap-1 p-3.5 rounded-2xl bg-black/40 border border-white/[0.08]">
      {STRAND_COLORS.map((hex, i) => (
        <div
          key={i}
          className="w-[5px] rounded-full"
          style={{
            height: 30,
            backgroundColor: hex,
            animation: `residue-pulse 1.1s ease-in-out ${i * 0.08}s infinite`,
            boxShadow: `0 0 8px ${hex}80`,
          }}
        />
      ))}
    </div>
    <div className="flex flex-col items-center gap-1">
      <span
        className="text-xs font-bold text-slate-300 tracking-widest uppercase font-mono"
      >
        PREDICTING STRUCTURE
      </span>
      <span className="text-[10px] font-mono text-slate-500 tracking-wider">
        SYS_INFERENCE // ASSEMBLING_RIBBON
      </span>
    </div>
  </div>
);

export default AssemblyLoader;
