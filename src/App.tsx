import { useEffect, useRef, useState } from 'react'
import {
	ACESFilmicToneMapping,
	Color,
	DirectionalLight,
	HemisphereLight,
	MathUtils,
	Mesh,
	Object3D,
	PerspectiveCamera,
	RenderPipeline,
	SRGBColorSpace,
	Scene,
	Texture,
	Vector2,
	Vector3,
	WebGPURenderer,
	type Material,
} from 'three/webgpu'
import { pass } from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { SSF } from './ssf/SSF.ts'

/**
 * 屏幕空间雾演示。
 *
 * 渲染分两段：
 * 1. pass(scene, camera) 把场景画进带深度的半精度目标。这一步不做色调映射。
 * 2. SSF 的 NodeMaterial 在全屏四边形上采样颜色和深度，在线性空间里混合雾。
 * RenderPipeline 默认 outputColorTransform 为开，会用 renderOutput 包住雾节点。
 * renderOutput 内部调用 .toneMapping()，创建 ToneMappingNode，按 renderer.toneMapping 做 ACES，再转到 sRGB。
 * 画全屏四边形时，管线会暂时把渲染器的 toneMapping 设成 NoToneMapping，避免做两次。
 *
 * 右侧参数只改雾的 uniform，不重建场景，也不重建节点图。
 */

/** 模型在场景里大约 550 米见方，路面高度大约 132 到 173。机位站在高处土路，朝低处看。 */
const MODEL_URL = '/models/forest/scene.gltf'
const EYE = new Vector3( 72, 178, 55 )
const LOOK = new Vector3( -70, 148, -45 )
/** 方向光放在观察点的反方向上。距离只影响光照衰减，不影响雾的方向。雾读的是传播方向向量。 */
const SUN_DISTANCE = 280
/** WASD 的水平移动速度，单位是场景单位每秒。这条路大约几百米长。 */
const MOVE_SPEED = 48
const MOVE_KEYS = new Set( [ 'KeyW', 'KeyA', 'KeyS', 'KeyD' ] )
/**
 * 散射颜色不超过 1。ToneMappingNode 的 ACES 会把光晕和雾色压到一起。
 * 强度 4 只在开着色调映射时使用。关掉色调映射要改回 1，否则光晕被截成白块。
 */
const INSCATTERING_INTENSITY = 4

type Settings = {
	enabled: boolean
	fogColor: string
	fogMaxOpacity: number
	fogDensity: number
	fogHeight: number
	fogHeightFalloff: number
	fogStartDistance: number
	noiseStrength: number
	noiseScale: number
	noiseSpeedX: number
	noiseSpeedZ: number
	azimuth: number
	elevation: number
	inscatteringColor: string
	inscatteringExponent: number
	inscatteringStart: number
}

/**
 * 林间晨雾。雾面放在路面低处附近（约 145），高处的树冠会露出。
 * 密度按这条路的尺度调过：效果默认 0.02 / 雾高 10，在几百米的场景里几乎看不见。
 */
const morning: Settings = {
	enabled: true,
	fogColor: '#c9d4cc',
	fogMaxOpacity: 0.9,
	fogDensity: 0.012,
	fogHeight: 145,
	fogHeightFalloff: 0.055,
	fogStartDistance: 18,
	noiseStrength: 8,
	noiseScale: 0.012,
	noiseSpeedX: 0.25,
	noiseSpeedZ: 0.08,
	azimuth: 30,
	elevation: 25,
	inscatteringColor: '#ffe2b0',
	inscatteringExponent: 5,
	inscatteringStart: 40,
}

const presets: { name: string, settings: Settings }[] = [
	{ name: '林间晨雾', settings: morning },
	{
		name: '贴地薄雾',
		settings: {
			...morning,
			fogMaxOpacity: 0.75,
			fogDensity: 0.01,
			fogHeight: 136,
			fogHeightFalloff: 0.14,
			fogStartDistance: 8,
			noiseStrength: 4,
			inscatteringExponent: 8,
			elevation: 35,
		},
	},
	{
		name: '黄昏光晕',
		settings: {
			...morning,
			fogColor: '#cbb59a',
			fogMaxOpacity: 0.95,
			fogDensity: 0.014,
			fogHeight: 155,
			fogHeightFalloff: 0.04,
			noiseStrength: 10,
			azimuth: 35,
			elevation: 10,
			inscatteringColor: '#ffb15a',
			inscatteringExponent: 3,
		},
	},
]

/**
 * 太阳方位和仰角换成光的传播方向。
 * 方位 0 朝 +Z 传播。仰角 0 是地平线，增大则 Y 分量为负（从上往下照）。
 * 朝向光源看时，视线的反方向和这个向量对齐，雾里的散射最亮。
 */
function sunDirection( azimuth: number, elevation: number ) {

	const az = MathUtils.degToRad( azimuth )
	const el = MathUtils.degToRad( elevation )
	const horizontal = Math.cos( el )
	return new Vector3( Math.sin( az ) * horizontal, -Math.sin( el ), Math.cos( az ) * horizontal )

}

/**
 * 把面板参数写进雾效、背景和方向光。
 * 背景用雾色，天空和浓雾接在一起。方向光的位置在观察点沿传播方向的反方向上。
 * 散射颜色先乘强度再写入。Color.set 会把十六进制从 sRGB 转到线性空间，乘 4 发生在线性值上。
 */
function applySettings( effect: SSF, scene: Scene, sun: DirectionalLight, settings: Settings ) {

	effect.enabled = settings.enabled
	effect.fogColor = settings.fogColor
	effect.fogMaxOpacity = settings.fogMaxOpacity
	effect.fogDensity = settings.fogDensity
	effect.fogHeight = settings.fogHeight
	effect.fogHeightFalloff = settings.fogHeightFalloff
	effect.fogStartDistance = settings.fogStartDistance
	effect.noiseStrength = settings.noiseStrength
	effect.noiseScale = settings.noiseScale
	effect.noiseSpeed = new Vector2( settings.noiseSpeedX, settings.noiseSpeedZ )
	effect.directionalInscatteringColor = new Color( settings.inscatteringColor ).multiplyScalar( INSCATTERING_INTENSITY )
	effect.directionalInscatteringExponent = settings.inscatteringExponent
	effect.directionalInscatteringStartDistance = settings.inscatteringStart
	const direction = sunDirection( settings.azimuth, settings.elevation )
	effect.inscatteringLightDirection = direction
	scene.background = new Color( settings.fogColor )
	sun.position.copy( LOOK ).addScaledVector( direction, -SUN_DISTANCE )
	sun.target.position.copy( LOOK )
	sun.target.updateMatrixWorld()

}

// 叶片是 BLEND，GLTFLoader 会关掉 depthWrite。雾读到的是叶片后面的远景深度，树冠会被整片涂成雾。
const foliageMaterials = new Set( [
	'Background_Tree_Atlas',
	'Forest_Bush',
	'Grass_Vegetation_Dry',
	'Grass_Vegetation_Green',
] )

function cutoutFoliage( root: Object3D ) {

	root.traverse( ( obj ) => {

		const mesh = obj as Mesh
		if ( !mesh.isMesh ) return
		const materials = ( Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) as Material[]
		for ( const material of materials ) {

			if ( !foliageMaterials.has( material.name ) ) continue
			material.transparent = false
			material.depthWrite = true
			material.alphaTest = 0.4

		}

	} )

}

// 土路、车辙、水坑的高光在掠射角下会整片变成镜面。去掉高光，并让可见贴图像素写入深度，雾不再用路面背后的远景。
const glossyRoadMaterials = new Set( [
	'Dirt_Road',
	'Dirt_Road_Trails',
	'Puddle_Streaks',
	'Road_Edge_Gravel_Dusty',
] )

function dullRoad( root: Object3D ) {

	root.traverse( ( obj ) => {

		const mesh = obj as Mesh
		if ( !mesh.isMesh ) return
		const materials = ( Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) as Material[]
		for ( const material of materials ) {

			if ( !glossyRoadMaterials.has( material.name ) ) continue
			const surface = material as Material & {
				metalness: number
				roughness: number
				roughnessMap: Texture | null
				metalnessMap: Texture | null
				specularIntensity: number
				specularIntensityMap: Texture | null
			}
			surface.metalness = 0
			surface.roughness = 1
			surface.roughnessMap = null
			surface.metalnessMap = null
			surface.specularIntensity = 0
			surface.specularIntensityMap = null
			// 这些贴图是叠在地形上的透明贴花，alpha 只有大约 0.4，会直接看到路面底下的土和树叶。
			material.transparent = false
			material.depthWrite = true
			material.alphaTest = material.name === 'Puddle_Streaks' ? 0.35 : 0.02
			// 贴花和路面共面。偏移避免和底面抢深度
			if ( material.name !== 'Dirt_Road' ) {

				material.polygonOffset = true
				material.polygonOffsetFactor = -2
				material.polygonOffsetUnits = -2

			}

		}

	} )

}

/** 卸掉网格、材质和贴图。React StrictMode 会装卸载各一次，不释放的话模型会加载两份。 */
function disposeObject( root: Object3D ) {

	root.traverse( ( obj ) => {

		const mesh = obj as Mesh
		if ( !mesh.isMesh ) return
		mesh.geometry?.dispose()
		const materials = ( Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) as Material[]
		for ( const material of materials ) {

			for ( const value of Object.values( material ) ) {

				if ( value instanceof Texture ) value.dispose()

			}
			material.dispose()

		}

	} )

}

function Slider( {
	label,
	min,
	max,
	step,
	value,
	onChange,
}: {
	label: string
	min: number
	max: number
	step: number
	value: number
	onChange: ( value: number ) => void
} ) {

	const digits = step < 0.01 ? 3 : step < 1 ? 2 : 0
	return (
		<label className="field">
			<span>{label}<b>{value.toFixed( digits )}</b></span>
			<input
				type="range"
				min={min}
				max={max}
				step={step}
				value={value}
				onChange={( event ) => onChange( Number( event.target.value ) )}
			/>
		</label>
	)

}

export default function App() {

	const hostRef = useRef<HTMLDivElement>( null )
	const settingsRef = useRef( morning )
	const effectRef = useRef<SSF | null>( null )
	const sceneRef = useRef<Scene | null>( null )
	const sunRef = useRef<DirectionalLight | null>( null )
	const [ settings, setSettings ] = useState( morning )
	const [ status, setStatus ] = useState<'loading' | 'ready' | 'error'>( 'loading' )
	const [ progress, setProgress ] = useState( 0 )
	const [ panelOpen, setPanelOpen ] = useState( true )
	const [ gpuError, setGpuError ] = useState<string | null>( null )

	// 滑块回调读的是这一帧的 settings。场景 effect 不能依赖 settings，否则拖动滑块会拆掉 WebGPU 上下文
	settingsRef.current = settings

	// 场景只创建一次。
	useEffect( () => {

		const host = hostRef.current
		if ( !host ) return

		const canvas = document.createElement( 'canvas' )
		host.appendChild( canvas )

		// alpha 默认是 true，画布按预乘透明合成。全屏输出用不透明画布，避免 alpha 为 0 时整张图被合成掉
		const renderer = new WebGPURenderer( { canvas, antialias: false, alpha: false, powerPreference: 'high-performance' } )
		renderer.outputColorSpace = SRGBColorSpace
		// 这个值不会在画场景时生效。pass 画到离屏目标，RenderPipeline 画全屏时也会暂时关掉它
		// 它被读进 renderOutput → ToneMappingNode，在雾节点之后、写入画布之前做 ACES
		renderer.toneMapping = ACESFilmicToneMapping
		renderer.setPixelRatio( Math.min( window.devicePixelRatio, 2 ) )

		const scene = new Scene()
		const camera = new PerspectiveCamera( 50, 1, 0.3, 3000 )
		camera.position.copy( EYE )

		const sun = new DirectionalLight( '#fff1d6', 2.6 )
		sun.target.position.copy( LOOK )
		scene.add( sun, sun.target )
		scene.add( new HemisphereLight( '#e7eef2', '#3e4a32', 0.55 ) )

		// pass 在全屏材质采样它的纹理时才渲染场景，并带上深度附件。目标类型跟随渲染器，默认半精度，能记下大于 1 的颜色
		const scenePass = pass( scene, camera )
		const effect = new SSF( scenePass.getTextureNode( 'output' ), scenePass.getTextureNode( 'depth' ) )
		effect.setCamera( camera )
		// outputNode 是线性空间的雾颜色。管线再包 ToneMappingNode，不要在 SSF 里再做一次
		const pipeline = new RenderPipeline( renderer, effect.outputNode )

		const controls = new OrbitControls( camera, canvas )
		controls.target.copy( LOOK )
		controls.enableDamping = true
		controls.maxDistance = 450
		controls.minDistance = 2
		controls.update()

		// W/S 沿相机视线前进后退，A/D 沿画面左右平移。相机和观察点一起动，朝向不变。焦点在输入框里时不响应
		let lastTime = performance.now()
		const pressed = new Set<string>()
		const forward = new Vector3()
		const right = new Vector3()
		const move = new Vector3()
		const onKeyDown = ( event: KeyboardEvent ) => {

			if ( !MOVE_KEYS.has( event.code ) ) return
			if ( event.target instanceof HTMLElement && event.target.closest( 'input, textarea, select' ) ) return
			pressed.add( event.code )
			event.preventDefault()

		}
		const onKeyUp = ( event: KeyboardEvent ) => {

			pressed.delete( event.code )

		}
		const onBlur = () => {

			pressed.clear()

		}
		window.addEventListener( 'keydown', onKeyDown )
		window.addEventListener( 'keyup', onKeyUp )
		window.addEventListener( 'blur', onBlur )

		sceneRef.current = scene
		sunRef.current = sun
		effectRef.current = effect
		applySettings( effect, scene, sun, settingsRef.current )

		const resize = () => {

			const width = host.clientWidth
			const height = host.clientHeight
			if ( width === 0 || height === 0 ) return
			camera.aspect = width / height
			camera.updateProjectionMatrix()
			// pass 每帧按 drawing buffer 调整目标尺寸，这里只改渲染器
			renderer.setSize( width, height, false )

		}
		resize()
		const observer = new ResizeObserver( resize )
		observer.observe( host )

		let disposed = false
		new GLTFLoader().load( MODEL_URL, ( gltf ) => {

			if ( disposed ) {

				disposeObject( gltf.scene )
				return

			}
			cutoutFoliage( gltf.scene )
			dullRoad( gltf.scene )
			scene.add( gltf.scene )
			setStatus( 'ready' )

		}, ( event ) => {

			if ( !disposed && event.total > 0 ) setProgress( event.loaded / event.total )

		}, () => {

			if ( !disposed ) setStatus( 'error' )

		} )

		renderer.init().then( () => {

			if ( disposed ) return
			renderer.setAnimationLoop( () => {

				const now = performance.now()
				const delta = Math.min( ( now - lastTime ) / 1000, 0.05 )
				lastTime = now
				if ( pressed.size > 0 ) {

					camera.getWorldDirection( forward )
					right.setFromMatrixColumn( camera.matrixWorld, 0 )
					move.set( 0, 0, 0 )
					if ( pressed.has( 'KeyW' ) ) move.add( forward )
					if ( pressed.has( 'KeyS' ) ) move.sub( forward )
					if ( pressed.has( 'KeyD' ) ) move.add( right )
					if ( pressed.has( 'KeyA' ) ) move.sub( right )
					if ( move.lengthSq() > 0 ) {

						move.normalize().multiplyScalar( MOVE_SPEED * delta )
						camera.position.add( move )
						controls.target.add( move )

					}

				}
				controls.update()
				// 不要调用 renderer.render。那会跳过雾和 ToneMappingNode，直接把场景画到画布
				pipeline.render()

			} )

		} ).catch( () => {

			if ( !disposed ) setGpuError( '当前浏览器不支持 WebGPU' )

		} )

		return () => {

			disposed = true
			renderer.setAnimationLoop( null )
			observer.disconnect()
			window.removeEventListener( 'keydown', onKeyDown )
			window.removeEventListener( 'keyup', onKeyUp )
			window.removeEventListener( 'blur', onBlur )
			controls.dispose()
			pipeline.dispose()
			effect.material.dispose()
			disposeObject( scene )
			renderer.dispose()
			canvas.remove()
			effectRef.current = null
			sceneRef.current = null
			sunRef.current = null

		}

	}, [] )

	// 滑块只更新 uniform、背景和方向光。
	useEffect( () => {

		const effect = effectRef.current
		const scene = sceneRef.current
		const sun = sunRef.current
		if ( !effect || !scene || !sun ) return
		applySettings( effect, scene, sun, settings )

	}, [ settings ] )

	const patch = ( partial: Partial<Settings> ) => {

		setSettings( ( current ) => ( { ...current, ...partial } ) )

	}

	return (
		<div className="app">
			<div className="view" ref={hostRef}>
				{( status !== 'ready' || gpuError ) && (
					<div className="status">
						{gpuError ?? ( status === 'error' ? '模型加载失败' : `正在加载林间土路 ${Math.round( progress * 100 )}%` )}
					</div>
				)}
				{!panelOpen && (
					<button type="button" className="panel-toggle" aria-expanded={false} onClick={() => setPanelOpen( true )}>
						参数
					</button>
				)}
			</div>
			{panelOpen && (
				<aside className="panel">
					<header className="panel-head">
						<div>
							<h1>屏幕空间雾</h1>
							<p>沿土路看：近处清晰，低处浓、高处淡，雾面起伏，朝向太阳时雾里发亮。WASD 沿相机视线前后左右移动。</p>
						</div>
						<button type="button" className="collapse" aria-expanded={panelOpen} onClick={() => setPanelOpen( false )}>
							收起
						</button>
					</header>
					<label className="toggle">
						<input
							type="checkbox"
							checked={settings.enabled}
							onChange={( event ) => patch( { enabled: event.target.checked } )}
						/>
						启用雾
					</label>
					<div className="presets">
						{presets.map( ( preset ) => (
							<button key={preset.name} type="button" onClick={() => setSettings( preset.settings )}>
								{preset.name}
							</button>
						) )}
					</div>
					<section>
						<h2>雾体</h2>
						<label className="field">
							<span>颜色</span>
							<input type="color" value={settings.fogColor} onChange={( event ) => patch( { fogColor: event.target.value } )} />
						</label>
						<Slider label="最大不透明度" min={0} max={1} step={0.01} value={settings.fogMaxOpacity} onChange={( fogMaxOpacity ) => patch( { fogMaxOpacity } )} />
						<Slider label="密度" min={0} max={0.04} step={0.001} value={settings.fogDensity} onChange={( fogDensity ) => patch( { fogDensity } )} />
						<Slider label="起始距离" min={0} max={200} step={1} value={settings.fogStartDistance} onChange={( fogStartDistance ) => patch( { fogStartDistance } )} />
					</section>
					<section>
						<h2>高度</h2>
						<Slider label="雾面高度" min={100} max={200} step={1} value={settings.fogHeight} onChange={( fogHeight ) => patch( { fogHeight } )} />
						<Slider label="高度衰减" min={0.01} max={0.2} step={0.005} value={settings.fogHeightFalloff} onChange={( fogHeightFalloff ) => patch( { fogHeightFalloff } )} />
					</section>
					<section>
						<h2>噪声</h2>
						<Slider label="强度" min={0} max={30} step={0.5} value={settings.noiseStrength} onChange={( noiseStrength ) => patch( { noiseStrength } )} />
						<Slider label="尺度" min={0.001} max={0.04} step={0.001} value={settings.noiseScale} onChange={( noiseScale ) => patch( { noiseScale } )} />
						<Slider label="速度 X" min={0} max={1} step={0.01} value={settings.noiseSpeedX} onChange={( noiseSpeedX ) => patch( { noiseSpeedX } )} />
						<Slider label="速度 Z" min={0} max={1} step={0.01} value={settings.noiseSpeedZ} onChange={( noiseSpeedZ ) => patch( { noiseSpeedZ } )} />
					</section>
					<section>
						<h2>散射</h2>
						<label className="field">
							<span>颜色</span>
							<input type="color" value={settings.inscatteringColor} onChange={( event ) => patch( { inscatteringColor: event.target.value } )} />
						</label>
						<Slider label="指数" min={1} max={24} step={0.5} value={settings.inscatteringExponent} onChange={( inscatteringExponent ) => patch( { inscatteringExponent } )} />
						<Slider label="起始距离" min={0} max={200} step={1} value={settings.inscatteringStart} onChange={( inscatteringStart ) => patch( { inscatteringStart } )} />
						<Slider label="太阳方位" min={-180} max={180} step={1} value={settings.azimuth} onChange={( azimuth ) => patch( { azimuth } )} />
						<Slider label="太阳仰角" min={0} max={80} step={1} value={settings.elevation} onChange={( elevation ) => patch( { elevation } )} />
					</section>
					<p className="credit">
						This work is based on "[UPDATE] Dirt Road Through Forest" (<a href="https://sketchfab.com/3d-models/update-dirt-road-through-forest-c4676cdf7715484382400ff63faffd45">https://sketchfab.com/3d-models/update-dirt-road-through-forest-c4676cdf7715484382400ff63faffd45</a>) by 99.Miles (<a href="https://sketchfab.com/99.Miles">https://sketchfab.com/99.Miles</a>) licensed under CC-BY-4.0 (<a href="http://creativecommons.org/licenses/by/4.0/">http://creativecommons.org/licenses/by/4.0/</a>)
					</p>
				</aside>
			)}
		</div>
	)

}
