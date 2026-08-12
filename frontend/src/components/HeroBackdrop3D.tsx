import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * HeroBackdrop3D — Ambient slow-rotating abstract protein helix behind the Dashboard hero section.
 *
 * - Very low opacity (20%) so it never competes with hero text
 * - Slow rotation (~70s per revolution)
 * - pointer-events: none on the wrapper so buttons remain clickable
 * - Respects prefers-reduced-motion (pauses rotation)
 * - Lightweight: single low-poly TorusKnot (~128 tubular segments), no shadows, no postprocessing
 */

const HelixMesh: React.FC<{ reducedMotion: boolean }> = ({ reducedMotion }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((_state, delta) => {
    if (!meshRef.current || reducedMotion) return;
    meshRef.current.rotation.y += delta * 0.09; // ~70s per revolution
    meshRef.current.rotation.x += delta * 0.015;
  });

  return (
    <mesh ref={meshRef} position={[3, 0, -2]}>
      <torusKnotGeometry args={[3.5, 0.6, 128, 16, 2, 3]} />
      <meshStandardMaterial
        color="#7B2FF7"
        emissive="#00D9C0"
        emissiveIntensity={0.3}
        transparent
        opacity={0.2}
        depthWrite={false}
        roughness={0.5}
        metalness={0.3}
        wireframe
      />
    </mesh>
  );
};

export const HeroBackdrop3D: React.FC = () => {
  const reducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  return (
    <div
      className="absolute inset-0 z-0 overflow-hidden"
      style={{ pointerEvents: 'none' }}
      aria-hidden="true"
    >
      <Canvas
        gl={{ antialias: false, powerPreference: 'low-power', alpha: true }}
        camera={{ fov: 50, position: [0, 0, 12], near: 0.1, far: 50 }}
        style={{ background: 'transparent' }}
        frameloop={reducedMotion ? 'demand' : 'always'}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.6} color="#C4B5FD" />
        <HelixMesh reducedMotion={reducedMotion} />
      </Canvas>
    </div>
  );
};

export default HeroBackdrop3D;
