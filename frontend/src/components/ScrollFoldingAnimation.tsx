import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { generateProteinPath } from '../utils/pathGenerator';
import { MODEL_METRICS } from '../config/modelMetrics';
import { STRUCTURE_COLORS } from '../utils/colors';

/**
 * ScrollFoldingAnimation — Scroll-driven protein folding animation for the Dashboard landing page.
 *
 * As the user scrolls through a full-viewport section, a 3D protein ribbon transitions
 * from an extended/unfolded chain into its folded secondary structure form.
 *
 * - Scroll position directly drives animation progress (not time-based autoplay)
 * - Uses IntersectionObserver + scroll position for scroll tracking
 * - Reuses existing generateProteinPath() for consistent ribbon geometry
 * - Static fallback for prefers-reduced-motion or WebGL failure
 */

interface FoldingRibbonProps {
  progress: number;
  reducedMotion: boolean;
}

const FoldingRibbon: React.FC<FoldingRibbonProps> = ({ progress, reducedMotion }) => {
  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);

  const pathData = useMemo(() => {
    return generateProteinPath(
      MODEL_METRICS.demoSequence,
      MODEL_METRICS.demoQ3Prediction
    );
  }, []);

  const L = MODEL_METRICS.demoSequence.length;
  const limit = Math.max(0, L - 1);

  // Pre-allocated temporaries
  const tempMatrix = useMemo(() => new THREE.Matrix4(), []);
  const tempPos = useMemo(() => new THREE.Vector3(), []);
  const tempDir = useMemo(() => new THREE.Vector3(), []);
  const tempQuat = useMemo(() => new THREE.Quaternion(), []);
  const tempScale = useMemo(() => new THREE.Vector3(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);
  const tempP1 = useMemo(() => new THREE.Vector3(), []);
  const tempP2 = useMemo(() => new THREE.Vector3(), []);
  const zAxis = useMemo(() => new THREE.Vector3(0, 0, 1), []);

  useFrame(() => {
    const mesh = instancedMeshRef.current;
    if (!mesh || limit === 0) return;

    const { foldedPoints, unfoldedPoints } = pathData;
    const p = reducedMotion ? 1 : progress;

    for (let i = 0; i < limit; i++) {
      const p1f = unfoldedPoints[i];
      const p2f = unfoldedPoints[i + 1];
      const p1t = foldedPoints[i];
      const p2t = foldedPoints[i + 1];
      if (!p1f || !p2f || !p1t || !p2t) continue;

      tempP1.lerpVectors(p1f, p1t, p);
      tempP2.lerpVectors(p2f, p2t, p);

      tempPos.addVectors(tempP1, tempP2).multiplyScalar(0.5);
      tempDir.subVectors(tempP2, tempP1);
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

      tempScale.set(w, h, len || 0.01);
      tempMatrix.compose(tempPos, tempQuat, tempScale);
      mesh.setMatrixAt(i, tempMatrix);

      // Color: lerp from neutral gray to structure color based on progress
      const key = (q3 === 'H' || q3 === 'E' || q3 === 'C') ? q3 : 'C';
      const targetHex = STRUCTURE_COLORS[key].hex;
      tempColor.set('#475569').lerp(new THREE.Color(targetHex), p);
      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={instancedMeshRef}
      args={[null as any, null as any, limit]}
      castShadow={false}
      receiveShadow={false}
    >
      <cylinderGeometry args={[0.5, 0.5, 1, 8]} />
      <meshStandardMaterial
        roughness={0.35}
        metalness={0.15}
        emissive="#1e1b4b"
        emissiveIntensity={0.2}
      />
    </instancedMesh>
  );
};

export const ScrollFoldingAnimation: React.FC = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [progress, setProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [webglFailed, setWebglFailed] = useState(false);

  const reducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  // IntersectionObserver to detect when section is in viewport
  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.05 }
    );
    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  // Scroll position → progress mapping
  useEffect(() => {
    if (reducedMotion || !isVisible) return;

    const handleScroll = () => {
      const section = sectionRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const viewH = window.innerHeight;

      // progress 0 when section top enters viewport bottom,
      // progress 1 when section bottom leaves viewport top
      const totalTravel = rect.height + viewH;
      const traveled = viewH - rect.top;
      const p = Math.max(0, Math.min(1, traveled / totalTravel));
      setProgress(p);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // initial
    return () => window.removeEventListener('scroll', handleScroll);
  }, [isVisible, reducedMotion]);

  // WebGL availability check
  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) setWebglFailed(true);
    } catch {
      setWebglFailed(true);
    }
  }, []);

  const showStatic = reducedMotion || webglFailed;

  return (
    <section
      ref={sectionRef}
      className="relative w-full overflow-hidden"
      style={{ height: '70vh', minHeight: 400 }}
    >
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[var(--aurora-violet)]/[0.03] to-transparent" />

      {/* Label */}
      <div className="absolute top-6 left-0 right-0 z-10 flex flex-col items-center gap-1 pointer-events-none">
        <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-widest">
          SCROLL_ANIMATION // PROTEIN_FOLDING
        </span>
        <span className="text-xs text-[var(--text-secondary)] font-medium">
          {showStatic ? 'Folded secondary structure' : `Folding progress: ${(progress * 100).toFixed(0)}%`}
        </span>
      </div>

      {showStatic ? (
        /* Static fallback: show a simple text-based representation */
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="flex flex-wrap gap-1 justify-center p-4 max-w-md">
              {MODEL_METRICS.demoQ3Prediction.map((q3, i) => (
                <span
                  key={i}
                  className="w-5 h-5 rounded-sm text-[8px] flex items-center justify-center font-mono font-bold"
                  style={{
                    backgroundColor: q3 === 'H' ? 'rgba(139,92,246,0.3)' : q3 === 'E' ? 'rgba(0,229,204,0.3)' : 'rgba(148,163,184,0.2)',
                    color: q3 === 'H' ? '#C4B5FD' : q3 === 'E' ? '#5EEAD4' : '#94A3B8',
                  }}
                >
                  {q3}
                </span>
              ))}
            </div>
            <p className="text-[10px] text-[var(--text-muted)] font-mono mt-2">
              Demo: {MODEL_METRICS.demoSequence.substring(0, 20)}...
            </p>
          </div>
        </div>
      ) : (
        <Canvas
          gl={{ antialias: true, powerPreference: 'high-performance', alpha: true }}
          camera={{ fov: 40, position: [0, 8, 35], near: 0.1, far: 200 }}
          style={{ background: 'transparent' }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000000, 0);
          }}
        >
          <ambientLight intensity={0.8} />
          <directionalLight position={[15, 20, 20]} intensity={1.2} color="#F8FAFC" />
          <directionalLight position={[-10, 10, 15]} intensity={0.6} color="#C4B5FD" />
          <pointLight position={[0, -15, 0]} intensity={0.5} color="#00E5CC" />

          <FoldingRibbon progress={progress} reducedMotion={reducedMotion} />
        </Canvas>
      )}

      {/* Scroll indicator */}
      {!showStatic && progress < 0.9 && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-1 pointer-events-none animate-pulse">
          <span className="text-[10px] font-mono text-[var(--text-muted)]">↓ Scroll to fold ↓</span>
        </div>
      )}
    </section>
  );
};

export default ScrollFoldingAnimation;
