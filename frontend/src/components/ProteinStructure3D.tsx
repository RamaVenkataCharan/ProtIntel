import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { ProteinRibbon, type ViewMode3D } from './ProteinRibbon';
import { RotateCcw, Play, Pause, Ruler, Camera, X } from 'lucide-react';
import { useThemeStore } from '../store/useThemeStore';
import { generateProteinPath } from '../utils/pathGenerator';

interface ProteinStructure3DProps {
  sequence: string | null;
  q3Prediction: string[] | null;
  q8Prediction?: string[] | null;
  confidence: number[] | null;
  residueImportance?: number[] | null;
  hoveredIndex: number | null;
  isPredicting: boolean;
  onHoverResidue?: (idx: number | null) => void;
}

// CameraController handles smooth lerp targeting for click-to-inspect and cinematic tour
interface CameraControllerProps {
  targetPosition: THREE.Vector3 | null;
  isCinematicTour: boolean;
  foldedPoints: THREE.Vector3[];
  reducedMotion: boolean;
  isInteracting: React.MutableRefObject<boolean>;
}

const CameraController: React.FC<CameraControllerProps> = ({
  targetPosition,
  isCinematicTour,
  foldedPoints,
  reducedMotion,
  isInteracting,
}) => {
  const tourProgressRef = useRef(0);

  useFrame((state, delta) => {
    // 1. Cinematic Tour Mode
    if (isCinematicTour && foldedPoints.length > 0 && !reducedMotion && !isInteracting.current) {
      tourProgressRef.current = (tourProgressRef.current + delta * 0.1) % 1;
      const pointIdx = Math.floor(tourProgressRef.current * (foldedPoints.length - 1));
      const pt = foldedPoints[pointIdx] || foldedPoints[0];

      // Position camera slightly offset from backbone point
      const camPos = new THREE.Vector3(pt.x + 8, pt.y + 6, pt.z + 14);
      state.camera.position.lerp(camPos, delta * 3);
      state.camera.lookAt(pt);
      return;
    }

    // 2. Click-to-Inspect Focus Target
    if (targetPosition && !isInteracting.current) {
      const camPos = new THREE.Vector3(targetPosition.x, targetPosition.y + 4, targetPosition.z + 16);
      state.camera.position.lerp(camPos, delta * 4);
    }
  });

  return null;
};

// GroupRig handles slow idle ambient spin
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
      currentRotation.current += delta * 0.04;
      groupRef.current.rotation.y = currentRotation.current;
    }
  });

  return <group ref={groupRef}>{children}</group>;
};

export const ProteinStructure3D: React.FC<ProteinStructure3DProps> = ({
  sequence,
  q3Prediction,
  q8Prediction,
  confidence,
  residueImportance,
  hoveredIndex,
  isPredicting,
  onHoverResidue,
}) => {
  const { theme } = useThemeStore();
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [resetTrigger, setResetTrigger] = useState(0);
  const isInteracting = useRef(false);

  // Interactive 3D States
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode3D>('structure');
  const [colorMorph, setColorMorph] = useState(0); // 0 = Q3, 1 = Q8
  
  // Feature 5: Scrubber / Playback
  const [scrubberIndex, setScrubberIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState<1 | 2 | 4>(1);

  // Feature 7: Measurement tool
  const [isMeasuringMode, setIsMeasuringMode] = useState(false);
  const [measurePoints, setMeasurePoints] = useState<[number, number] | null>(null);

  // Feature 8: Cinematic tour
  const [isCinematicTour, setIsCinematicTour] = useState(false);

  // Detect reduced motion & mobile
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mediaQuery.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    const checkDevice = () => setIsMobile(window.innerWidth < 768 || navigator.maxTouchPoints > 1);
    checkDevice();
    window.addEventListener('resize', checkDevice);
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  // Compute backbone points for click-focus & measurement
  const pathData = useMemo(() => {
    if (!sequence || !q3Prediction) return null;
    return generateProteinPath(sequence, q3Prediction);
  }, [sequence, q3Prediction]);

  // Handle residue click inspect
  const handleSelectResidue = (idx: number | null) => {
    if (isMeasuringMode) {
      if (idx === null) return;
      if (!measurePoints) {
        setMeasurePoints([idx, idx]);
      } else {
        setMeasurePoints([measurePoints[0], idx]);
        setIsMeasuringMode(false);
      }
      return;
    }
    setSelectedIndex(idx);
    if (onHoverResidue) onHoverResidue(idx);
  };

  const handleHoverResidue = (idx: number | null) => {
    if (onHoverResidue) onHoverResidue(idx);
  };

  // Playback Auto-Advance
  useEffect(() => {
    if (!isPlaying || !sequence) return;
    const intervalTime = 300 / playSpeed;
    const timer = setInterval(() => {
      setScrubberIndex((prev) => {
        const next = (prev === null ? 0 : prev + 1);
        if (next >= sequence.length) {
          setIsPlaying(false);
          return 0;
        }
        return next;
      });
    }, intervalTime);
    return () => clearInterval(timer);
  }, [isPlaying, sequence, playSpeed]);

  const triggerReset = () => {
    setResetTrigger((prev) => prev + 1);
    setSelectedIndex(null);
    setScrubberIndex(null);
    setMeasurePoints(null);
    setIsCinematicTour(false);
  };

  const hasData = Boolean(sequence && q3Prediction && confidence);
  const L = sequence?.length || 0;

  // Selected Residue Target Position
  const selectedTargetPosition = useMemo(() => {
    const activeIdx = selectedIndex ?? scrubberIndex;
    if (activeIdx === null || !pathData || !pathData.foldedPoints[activeIdx]) return null;
    return pathData.foldedPoints[activeIdx];
  }, [selectedIndex, scrubberIndex, pathData]);

  // Measured distance calculation
  const measurementDistance = useMemo(() => {
    if (!measurePoints || !pathData) return null;
    const p1 = pathData.foldedPoints[measurePoints[0]];
    const p2 = pathData.foldedPoints[measurePoints[1]];
    if (!p1 || !p2) return null;
    return p1.distanceTo(p2);
  }, [measurePoints, pathData]);

  const activeResidueInfo = useMemo(() => {
    const idx = selectedIndex ?? hoveredIndex ?? scrubberIndex;
    if (idx === null || !sequence || !q3Prediction || !confidence) return null;
    return {
      index: idx,
      aa: sequence[idx],
      q3: q3Prediction[idx],
      q8: q8Prediction?.[idx] || q3Prediction[idx],
      conf: confidence[idx],
      importance: residueImportance?.[idx],
    };
  }, [selectedIndex, hoveredIndex, scrubberIndex, sequence, q3Prediction, q8Prediction, confidence, residueImportance]);

  return (
    <div className="w-full h-[440px] bg-[radial-gradient(circle_at_30%_20%,#1a0b2e_0%,#0a0e17_60%,#050508_100%)] border border-[var(--border-subtle)] rounded-3xl relative overflow-hidden flex flex-col group/canvas">
      
      {/* ── TOP TOOLBAR ─────────────────────────────────────────────── */}
      <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-auto">
        <div className="flex flex-wrap items-center gap-1.5 bg-black/60 backdrop-blur-md p-1.5 rounded-2xl border border-white/[0.08]">
          <button
            onClick={triggerReset}
            disabled={!hasData || isPredicting}
            className="px-2.5 py-1 rounded-xl text-xs font-bold text-slate-300 hover:text-white hover:bg-white/10 transition-all flex items-center gap-1 cursor-pointer disabled:opacity-30"
            title="Reset focus and animation"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset</span>
          </button>

          {/* View Mode Switcher (Structure vs XAI Heatmap) */}
          <div className="h-4 w-[1px] bg-white/10 mx-1" />
          <button
            onClick={() => setViewMode('structure')}
            className={`px-2.5 py-1 rounded-xl text-xs font-bold transition-all cursor-pointer ${
              viewMode === 'structure' ? 'bg-violet-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Structure Mode
          </button>
          <button
            onClick={() => setViewMode('xai')}
            disabled={!residueImportance}
            className={`px-2.5 py-1 rounded-xl text-xs font-bold transition-all cursor-pointer ${
              viewMode === 'xai' ? 'bg-amber-500 text-black shadow-lg' : 'text-slate-400 hover:text-slate-200 disabled:opacity-30'
            }`}
          >
            XAI Heatmap
          </button>

          {/* Q3 vs Q8 Morph Slider */}
          <div className="h-4 w-[1px] bg-white/10 mx-1" />
          <span className="text-[10px] font-mono text-slate-400 font-bold px-1">Q3</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={colorMorph}
            onChange={(e) => setColorMorph(parseFloat(e.target.value))}
            className="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400"
            title="Morph colors between Q3 and Q8"
          />
          <span className="text-[10px] font-mono text-slate-400 font-bold px-1">Q8</span>
        </div>

        {/* Action Tools: Measurement & Cinematic Tour */}
        <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-md p-1.5 rounded-2xl border border-white/[0.08]">
          <button
            onClick={() => {
              setIsMeasuringMode(!isMeasuringMode);
              if (!isMeasuringMode) setMeasurePoints(null);
            }}
            className={`px-2 py-1 rounded-xl text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
              isMeasuringMode ? 'bg-amber-500 text-black' : 'text-slate-300 hover:text-white hover:bg-white/10'
            }`}
            title="Measure schematic distance between 2 residues"
          >
            <Ruler className="h-3.5 w-3.5" />
            <span>Measure</span>
          </button>

          <button
            onClick={() => {
              setIsCinematicTour(!isCinematicTour);
              if (!isCinematicTour) setSelectedIndex(null);
            }}
            className={`px-2 py-1 rounded-xl text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
              isCinematicTour ? 'bg-violet-500 text-white' : 'text-slate-300 hover:text-white hover:bg-white/10'
            }`}
            title="Cinematic fly-through camera mode"
          >
            <Camera className="h-3.5 w-3.5" />
            <span>Tour</span>
          </button>
        </div>
      </div>

      {/* ── FLOATING HUD RESIDUE INSPECT CARD (Feature 1) ─────────── */}
      {activeResidueInfo && (
        <div className="absolute top-16 left-4 z-20 glass-instrument-card p-4 rounded-2xl max-w-xs animate-spring-up border border-teal-400/30 shadow-2xl">
          <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/10">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
              <span className="text-xs font-mono font-bold text-teal-300">
                RESIDUE_INSPECT // {activeResidueInfo.aa}{activeResidueInfo.index + 1}
              </span>
            </div>
            <button
              onClick={() => { setSelectedIndex(null); handleHoverResidue(null); }}
              className="text-slate-400 hover:text-white cursor-pointer"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Position</span>
              <span className="text-slate-200 font-bold">Index {activeResidueInfo.index + 1}</span>
            </div>
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Confidence</span>
              <span className="text-teal-400 font-bold">{(activeResidueInfo.conf * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Q3 Structure</span>
              <span className="text-violet-400 font-bold">{activeResidueInfo.q3}</span>
            </div>
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Q8 DSSP</span>
              <span className="text-amber-400 font-bold">{activeResidueInfo.q8}</span>
            </div>
            {activeResidueInfo.importance !== undefined && (
              <div className="col-span-2 pt-1 border-t border-white/5">
                <span className="text-[9px] text-slate-500 font-bold uppercase block">XAI Attribution Score</span>
                <span className="text-emerald-400 font-bold">{activeResidueInfo.importance.toFixed(4)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── MEASUREMENT HUD TAG (Feature 7 Honesty Disclaimer) ──────── */}
      {measurePoints && measurementDistance !== null && (
        <div className="absolute top-16 right-4 z-20 glass-instrument-card p-3 rounded-2xl animate-spring-up border border-amber-400/40">
          <div className="flex items-center justify-between gap-2 mb-1">
            <span className="text-[10px] font-mono font-bold text-amber-400">SCHEMATIC_MEASUREMENT</span>
            <button onClick={() => setMeasurePoints(null)} className="text-slate-400 hover:text-white">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <p className="text-sm font-bold font-mono text-white">
            Residues {measurePoints[0] + 1} ↔ {measurePoints[1] + 1}: <span className="text-amber-400">{measurementDistance.toFixed(2)} units</span>
          </p>
          <span className="text-[9px] text-amber-400/80 font-mono block mt-1">
            ⚠️ Illustrative ribbon distance (not physical Ångstroms)
          </span>
        </div>
      )}

      {/* ── THREE.JS CANVAS ─────────────────────────────────────────── */}
      <div className="w-full flex-1">
        <Canvas
          shadows
          gl={{ antialias: !isMobile, powerPreference: 'high-performance' }}
          camera={{ fov: 40, near: 0.1, far: 100, position: [0, 0, 25] }}
        >
          <color attach="background" args={[theme === 'dark' ? '#0a0e17' : '#f1f5f9']} />
          <fog attach="fog" args={[theme === 'dark' ? '#0a0e17' : '#f1f5f9', 15, 45]} />

          {/* 3-Point Scientific Lighting Rig */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[12, 20, 15]} intensity={1.4} castShadow />
          <pointLight position={[-12, -10, -10]} intensity={0.6} color="#00D9C0" />
          <directionalLight position={[0, 5, -20]} intensity={1.8} color="#7B2FF7" />

          {/* Ground Contact Shadow Disc */}
          <mesh position={[0, -4.5, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
            <circleGeometry args={[18, 32]} />
            <meshBasicMaterial color="#050811" transparent opacity={0.6} />
          </mesh>

          {/* Camera Controller */}
          <CameraController
            targetPosition={selectedTargetPosition}
            isCinematicTour={isCinematicTour}
            foldedPoints={pathData?.foldedPoints || []}
            reducedMotion={reducedMotion}
            isInteracting={isInteracting}
          />

          <OrbitControls
            enableDamping
            dampingFactor={0.05}
            maxDistance={80}
            minDistance={10}
            onStart={() => {
              isInteracting.current = true;
              if (isCinematicTour) setIsCinematicTour(false);
            }}
            onEnd={() => {
              isInteracting.current = false;
            }}
          />

          <GroupRig reducedMotion={reducedMotion} isInteracting={isInteracting}>
            {hasData && (
              <ProteinRibbon
                sequence={sequence!}
                q3Prediction={q3Prediction!}
                q8Prediction={q8Prediction}
                confidence={confidence!}
                residueImportance={residueImportance}
                hoveredIndex={hoveredIndex}
                selectedIndex={selectedIndex}
                scrubberIndex={scrubberIndex}
                measurePoints={measurePoints}
                viewMode={viewMode}
                colorMorph={colorMorph}
                resetTrigger={resetTrigger}
                reducedMotion={reducedMotion}
                isMobile={isMobile}
                onSelectResidue={handleSelectResidue}
                onHoverResidue={handleHoverResidue}
              />
            )}
          </GroupRig>
        </Canvas>
      </div>

      {/* ── TIMELINE SCRUBBER & PLAYBACK CONTROLS (Feature 5) ──────── */}
      {hasData && (
        <div className="px-4 py-2.5 bg-black/80 backdrop-blur-md border-t border-white/10 flex items-center justify-between gap-4 z-20 select-none">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1.5 rounded-xl bg-teal-500/20 border border-teal-500/40 text-teal-300 hover:bg-teal-500/30 transition-all cursor-pointer"
              title={isPlaying ? 'Pause auto-advance' : 'Play timeline auto-advance'}
            >
              {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </button>

            <button
              onClick={() => setPlaySpeed(playSpeed === 1 ? 2 : playSpeed === 2 ? 4 : 1)}
              className="px-2 py-0.5 rounded-lg bg-black/40 border border-white/10 text-[10px] font-mono font-bold text-amber-400 cursor-pointer"
            >
              {playSpeed}x
            </button>
          </div>

          <div className="flex-1 flex items-center gap-3">
            <span className="text-[10px] font-mono text-slate-400">1</span>
            <input
              type="range"
              min="0"
              max={L - 1}
              value={scrubberIndex ?? 0}
              onChange={(e) => {
                const idx = parseInt(e.target.value, 10);
                setScrubberIndex(idx);
                handleHoverResidue(idx);
              }}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400"
            />
            <span className="text-[10px] font-mono text-slate-400">{L}</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded-md font-mono text-[9px] font-bold text-amber-400 bg-amber-400/10 border border-amber-400/20">
              Schematic 3D Ribbon (Illustrative 2° structure)
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProteinStructure3D;
