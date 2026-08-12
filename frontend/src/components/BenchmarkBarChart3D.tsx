import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { BENCHMARK_CONFIGS, BENCHMARK_METRICS, type BenchmarkMetricKey } from '../data/benchmarkData';

/**
 * BenchmarkBarChart3D — Rotatable 3D bar chart for Q3/Q8/MCC benchmark comparisons.
 *
 * Uses ONLY real evaluated metric values from CB513 evaluation artifacts.
 * Bars colored per configuration, with billboard text labels that face the camera.
 */

const CONFIG_COLORS = ['#8B5CF6', '#00E5CC', '#F59E0B'];
const BAR_WIDTH = 0.8;
const BAR_DEPTH = 0.8;
const METRIC_SPACING = 3.5;
const CONFIG_SPACING = 1.8;
const MAX_HEIGHT = 8;

interface BarProps {
  position: [number, number, number];
  height: number;
  color: string;
  label: string;
  value: number;
}

const Bar: React.FC<BarProps> = ({ position, height, color, value }) => {
  const [hovered, setHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  const targetHeight = useRef(0);

  // Animate bar growth
  useFrame((_state, delta) => {
    if (!meshRef.current) return;
    targetHeight.current += (height - targetHeight.current) * Math.min(1, delta * 4);
    const h = Math.max(0.01, targetHeight.current);
    meshRef.current.scale.y = h;
    meshRef.current.position.y = h / 2;
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        castShadow
      >
        <boxGeometry args={[BAR_WIDTH, 1, BAR_DEPTH]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered ? 0.4 : 0.15}
          roughness={0.3}
          metalness={0.2}
          transparent
          opacity={hovered ? 1.0 : 0.85}
        />
      </mesh>

      {/* Value label on top of bar (billboard) */}
      <BillboardText
        position={[0, height + 0.5, 0]}
        text={value >= 1 ? value.toFixed(1) + '%' : value.toFixed(3)}
        fontSize={0.32}
        color={hovered ? '#F8FAFC' : '#94A3B8'}
        bold
      />
    </group>
  );
};

/** Text that always faces the camera */
const BillboardText: React.FC<{
  position: [number, number, number];
  text: string;
  fontSize: number;
  color: string;
  bold?: boolean;
}> = ({ position, text, fontSize, color, bold }) => {
  return (
    <Text
      position={position}
      fontSize={fontSize}
      color={color}
      anchorX="center"
      anchorY="middle"
      font={undefined}
      fontWeight={bold ? 'bold' : 'normal'}
    >
      {text}
    </Text>
  );
};

/** Grid floor and axis helpers */
const ChartAxes: React.FC = () => {
  const metrics = BENCHMARK_METRICS;
  const configs = BENCHMARK_CONFIGS;

  return (
    <group>
      {/* Floor grid */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[metrics.length * METRIC_SPACING + 2, configs.length * CONFIG_SPACING + 2]} />
        <meshStandardMaterial color="#0A0E17" transparent opacity={0.6} />
      </mesh>

      {/* Y-axis scale lines */}
      {[0.25, 0.5, 0.75, 1.0].map((v) => (
        <group key={v}>
          <mesh position={[-1.5, v * MAX_HEIGHT, 0]}>
            <boxGeometry args={[0.02, 0.02, metrics.length * METRIC_SPACING + 1]} />
            <meshBasicMaterial color="#334155" transparent opacity={0.4} />
          </mesh>
          <BillboardText
            position={[-2.2, v * MAX_HEIGHT, 0]}
            text={(v * 100).toFixed(0) + '%'}
            fontSize={0.25}
            color="#64748B"
          />
        </group>
      ))}

      {/* Metric labels (Z-axis) */}
      {metrics.map((m, mi) => (
        <BillboardText
          key={m.key}
          position={[
            (configs.length - 1) * CONFIG_SPACING / 2,
            -0.8,
            mi * METRIC_SPACING - (metrics.length - 1) * METRIC_SPACING / 2,
          ]}
          text={m.label}
          fontSize={0.35}
          color={m.color}
          bold
        />
      ))}

      {/* Config labels (X-axis) */}
      {configs.map((c, ci) => (
        <BillboardText
          key={c.label}
          position={[
            ci * CONFIG_SPACING - (configs.length - 1) * CONFIG_SPACING / 2,
            -0.8,
            (metrics.length - 1) * METRIC_SPACING / 2 + 1.5,
          ]}
          text={c.label}
          fontSize={0.28}
          color={CONFIG_COLORS[ci % CONFIG_COLORS.length]}
          bold
        />
      ))}
    </group>
  );
};

const ChartScene: React.FC = () => {
  const configs = BENCHMARK_CONFIGS;
  const metrics = BENCHMARK_METRICS;

  const bars = useMemo(() => {
    const result: BarProps[] = [];

    configs.forEach((config, ci) => {
      metrics.forEach((metric, mi) => {
        const rawValue = config[metric.key as BenchmarkMetricKey];
        const displayValue = metric.key === 'q3Mcc' ? rawValue : rawValue * 100;
        const barHeight = rawValue * MAX_HEIGHT; // normalized 0-1 to bar height

        result.push({
          position: [
            ci * CONFIG_SPACING - (configs.length - 1) * CONFIG_SPACING / 2,
            0,
            mi * METRIC_SPACING - (metrics.length - 1) * METRIC_SPACING / 2,
          ],
          height: barHeight,
          color: CONFIG_COLORS[ci % CONFIG_COLORS.length],
          label: `${config.label} ${metric.label}`,
          value: displayValue,
        });
      });
    });

    return result;
  }, []);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 15, 10]} intensity={1.2} castShadow />
      <directionalLight position={[-8, 10, -8]} intensity={0.5} color="#C4B5FD" />
      <pointLight position={[0, -5, 0]} intensity={0.3} color="#00E5CC" />

      <ChartAxes />
      {bars.map((bar, i) => (
        <Bar key={i} {...bar} />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        maxPolarAngle={Math.PI * 0.48}
        minPolarAngle={Math.PI * 0.1}
        maxDistance={30}
        minDistance={8}
      />
    </>
  );
};

export const BenchmarkBarChart3D: React.FC = () => {
  return (
    <div className="w-full h-[400px] rounded-2xl overflow-hidden border border-[var(--border-subtle)] bg-[radial-gradient(circle_at_50%_30%,#161b2e_0%,#090d18_70%,#04060b_100%)] relative">
      {/* Header */}
      <div className="absolute top-3 left-4 z-10 pointer-events-none">
        <span className="text-[10px] font-mono font-bold text-violet-400 uppercase tracking-widest">
          3D_BENCHMARK // CB513_VERIFIED
        </span>
      </div>

      {/* Source attribution */}
      <div className="absolute bottom-3 left-4 z-10 pointer-events-none">
        <span className="text-[9px] font-mono text-slate-600">
          Source: logs/evaluation/cb513_results.json · final_cb513_metrics.json
        </span>
      </div>

      <Canvas
        shadows
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ fov: 35, position: [8, 10, 12], near: 0.1, far: 100 }}
      >
        <color attach="background" args={['#070a12']} />
        <fog attach="fog" args={['#070a12', 25, 50]} />
        <ChartScene />
      </Canvas>
    </div>
  );
};

export default BenchmarkBarChart3D;
