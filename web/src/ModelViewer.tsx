import { Button, Tooltip } from "@heroui/react";
import { Box, Grid3X3, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

type ModelViewerProps = {
  modelUrl?: string;
  isWorking: boolean;
};

export default function ModelViewer({ modelUrl, isWorking }: ModelViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const resetRef = useRef<() => void>(() => undefined);
  const meshRef = useRef<THREE.Object3D | null>(null);
  const [wireframe, setWireframe] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#171a18");
    scene.fog = new THREE.Fog("#171a18", 9, 16);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 2.4;
    controls.maxDistance = 10;

    const resetCamera = () => {
      camera.position.set(3.6, 2.7, 4.2);
      controls.target.set(0, 0, 0);
      controls.update();
    };
    resetCamera();
    resetRef.current = resetCamera;

    scene.add(new THREE.HemisphereLight("#e7f4eb", "#4a4037", 2.1));
    const keyLight = new THREE.DirectionalLight("#fff4de", 3.2);
    keyLight.position.set(4, 6, 5);
    keyLight.castShadow = true;
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight("#9fc6ff", 1.4);
    rimLight.position.set(-4, 2, -3);
    scene.add(rimLight);

    const grid = new THREE.GridHelper(10, 20, "#69716b", "#2b302d");
    grid.position.y = -1.05;
    scene.add(grid);

    const base = new THREE.Mesh(
      new THREE.CircleGeometry(1.45, 64),
      new THREE.MeshStandardMaterial({ color: "#202522", roughness: 0.96 }),
    );
    base.rotation.x = -Math.PI / 2;
    base.position.y = -1.04;
    base.receiveShadow = true;
    scene.add(base);

    const resize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let animationFrame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    (container as HTMLDivElement & { scene?: THREE.Scene }).scene = scene;
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      delete (container as HTMLDivElement & { scene?: THREE.Scene }).scene;
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current as (HTMLDivElement & { scene?: THREE.Scene }) | null;
    const scene = container?.scene;
    if (!scene) return;
    if (meshRef.current) {
      const previous = meshRef.current;
      previous.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
          else child.material.dispose();
        }
      });
      scene.remove(previous);
      meshRef.current = null;
    }
    if (!modelUrl) {
      setLoading(false);
      return;
    }
    setLoading(true);
    const loader = new OBJLoader();
    loader.load(
      modelUrl,
      (object) => {
        const bounds = new THREE.Box3().setFromObject(object);
        const center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        const scale = 2 / Math.max(size.x, size.y, size.z, 0.001);
        object.position.sub(center);
        object.scale.setScalar(scale);
        object.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.material = new THREE.MeshStandardMaterial({
              color: "#a8dab7",
              roughness: 0.72,
              metalness: 0.04,
              flatShading: true,
              wireframe,
            });
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        scene.add(object);
        meshRef.current = object;
        resetRef.current();
        setLoading(false);
      },
      undefined,
      () => setLoading(false),
    );
  }, [modelUrl]);

  useEffect(() => {
    meshRef.current?.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshStandardMaterial) {
        child.material.wireframe = wireframe;
      }
    });
  }, [wireframe]);

  return (
    <div className="viewer-root">
      <div ref={containerRef} className="viewer-canvas" aria-label="三维模型预览" />
      <div className="viewer-toolbar">
        <Tooltip delay={0}>
          <Button isIconOnly aria-label="重置视角" variant="secondary" onPress={() => resetRef.current()}>
            <RotateCcw size={18} />
          </Button>
          <Tooltip.Content>重置视角</Tooltip.Content>
        </Tooltip>
        <Tooltip delay={0}>
          <Button
            isIconOnly
            aria-label="切换线框"
            variant={wireframe ? "primary" : "secondary"}
            onPress={() => setWireframe((value) => !value)}
          >
            <Grid3X3 size={18} />
          </Button>
          <Tooltip.Content>切换线框</Tooltip.Content>
        </Tooltip>
      </div>
      {!modelUrl && !isWorking ? (
        <div className="viewer-empty">
          <Box size={30} strokeWidth={1.5} />
          <span>等待模型</span>
        </div>
      ) : null}
      {isWorking || loading ? (
        <div className="viewer-loading">
          <span className="viewer-loading-dot" />
          {loading ? "载入网格" : "生成体素"}
        </div>
      ) : null}
      <div className="viewer-axis" aria-hidden="true">
        <span className="axis-x">X</span>
        <span className="axis-y">Y</span>
        <span className="axis-z">Z</span>
      </div>
    </div>
  );
}
