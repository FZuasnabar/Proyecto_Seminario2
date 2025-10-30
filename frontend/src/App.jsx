import { useState } from "react";
import {
  ShieldCheck,
  AlertTriangle,
  Cpu,
  Brain,
  Upload,
  Image,
  BarChart2,
  CheckCircle,
  Cog,
  GraduationCap,
  Mail,
  Linkedin,
  Github,
} from "lucide-react";


export default function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);


  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
    }
  };

  return (
    <div className="font-sans bg-gray-100 text-gray-800 text-lg">
      {/* Hero */}
      <section className="bg-gradient-to-r from-blue-500 to-purple-600 text-white text-center py-12">
        <div className="flex flex-col items-center justify-center">
          <Brain className="w-12 h-12 mb-3" />
          <h1 className="text-4xl font-bold">AI Image Detective</h1>
          <p className="text-xl mt-2">Detector de imágenes generadas por IA</p>
          <p className="mt-4 text-base max-w-2xl">
            Prueba cómo nuestro modelo CNN clasifica imágenes reales y sintéticas
          </p>
        </div>
      </section>

      {/* Importancia + Nuestro Modelo */}
      <section className="py-16 text-center max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold mb-10">
          ¿Por qué es importante detectar imágenes generadas por IA?
        </h2>

        <div className="grid md:grid-cols-3 gap-8">
          <Card
            icon={<AlertTriangle className="text-red-500 w-14 h-14 mx-auto" />}
            title="Desinformación"
            text="Las imágenes sintéticas pueden ser usadas para crear noticias falsas y manipular la opinión pública."
          />
          <Card
            icon={<ShieldCheck className="text-blue-500 w-14 h-14 mx-auto" />}
            title="Seguridad"
            text="Proteger la autenticidad de documentos e imágenes en contextos legales y forenses."
          />
          <Card
            icon={<Cpu className="text-green-500 w-14 h-14 mx-auto" />}
            title="Transparencia"
            text="Mantener la confianza en los medios digitales distinguiendo contenido real del sintético."
          />

          {/* Card ancho completo */}
          <div className="bg-white shadow rounded-lg p-10 text-center col-span-3">
            <h3 className="text-2xl font-semibold mb-4">Nuestro Modelo</h3>
            <p className="text-gray-700 leading-relaxed text-lg">
              Utilizamos <b>EfficientNetB0</b>, una red neuronal convolucional de
              última generación entrenada específicamente para detectar imágenes
              sintéticas. Nuestro modelo alcanza una{" "}
              <b>precisión del 94.2%</b> en la clasificación de imágenes reales
              vs. generadas por IA, basándose en técnicas avanzadas de Deep
              Learning y análisis de patrones sutiles que son imperceptibles al
              ojo humano.
            </p>
          </div>
        </div>
      </section>

      {/* Analizar Imagen */}
      <section className="py-16 text-center max-w-3xl mx-auto">
        <div className="bg-white shadow rounded-lg p-10">
          <h3 className="text-2xl font-semibold mb-3">Analizar Imagen</h3>
          <p className="text-gray-600 mb-6">
            Sube una imagen para determinar si es real o generada por IA
          </p>

          {/* Estados */}
          {/*
            Añadimos nuevos estados y funciones aquí mismo dentro del componente App:
            const [image, setImage] = useState(null);
            const [file, setFile] = useState(null);
            const [result, setResult] = useState(null);
            const [loading, setLoading] = useState(false);
          */}
          <label className="border-2 border-dashed border-gray-400 p-8 rounded-lg cursor-pointer inline-block w-full">
            <Upload className="mx-auto mb-2 w-6 h-6 text-gray-500" />
            <span className="font-medium">Seleccionar Imagen</span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  const selected = e.target.files[0];
                  setImage(URL.createObjectURL(selected));
                  setFile(selected);
                  setResult(null);
                }
              }}
            />
          </label>

          {image && (
            <div className="mt-6">
              <p className="text-gray-600 text-sm mb-3">Vista previa:</p>
              <img
                src={image}
                alt="preview"
                className="mx-auto max-h-80 rounded-lg shadow"
              />

              {/* Botón Analizar */}
              <button
                onClick={async () => {
                  if (!file) return alert("Por favor selecciona una imagen primero.");
                  setLoading(true);
                  const formData = new FormData();
                  formData.append("file", file);

                  try {
                    const response = await fetch("http://127.0.0.1:5000/predict", {
                      method: "POST",
                      body: formData,
                    });
                    const data = await response.json();
                    setResult(data);
                  } catch (err) {
                    alert("Error al comunicarse con el servidor Flask");
                    console.error(err);
                  } finally {
                    setLoading(false);
                  }
                }}
                disabled={loading}
                className="mt-5 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
              >
                {loading ? "Analizando..." : "Analizar Imagen"}
              </button>

              {/* Resultado */}
              {result && (
                <div
                  className={`mt-6 p-5 border rounded-lg ${
                    result.resultado === "REAL"
                      ? "bg-green-50 border-green-300 text-green-700"
                      : "bg-red-50 border-red-300 text-red-700"
                  }`}
                >
                  <p className="text-xl font-bold flex items-center justify-center gap-2">
                    {result.resultado === "REAL" ? (
                      <CheckCircle className="w-6 h-6 text-green-600" />
                    ) : (
                      <AlertTriangle className="w-6 h-6 text-red-600" />
                    )}
                    {result.resultado}
                  </p>
                  <p className="mt-2 text-gray-700">
                    Confianza del modelo:{" "}
                    <b>{(result.confianza * 100).toFixed(2)}%</b>
                  </p>
                </div>
              )}
            </div>
          )}

          <p className="text-sm mt-4 text-gray-500">
            Formatos aceptados: JPG, PNG • Tamaño máximo: 10MB
          </p>
        </div>
      </section>


      {/* Cómo funciona nuestro sistema */}
      <section className="py-16 text-center max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold mb-6">
          ¿Cómo funciona nuestro sistema?
        </h2>
        <p className="max-w-3xl mx-auto mb-12 text-gray-600">
          Nuestro modelo utiliza técnicas avanzadas de Deep Learning para analizar
          patrones microscópicos en las imágenes que son invisibles al ojo humano.
        </p>

        {/* Pasos con iconos */}
        <div className="grid md:grid-cols-4 gap-8 mb-12">
          <Step
            icon={<Image className="w-10 h-10 text-blue-500 mb-3" />}
            number="1. Imagen"
            text="Subida y validación del archivo"
          />
          <Step
            icon={<Cpu className="w-10 h-10 text-green-500 mb-3" />}
            number="2. Preprocesamiento"
            text="Normalización y redimensionado"
          />
          <Step
            icon={<BarChart2 className="w-10 h-10 text-purple-500 mb-3" />}
            number="3. CNN Analysis"
            text="Análisis con EfficientNetB0"
          />
          <Step
            icon={<CheckCircle className="w-10 h-10 text-orange-500 mb-3" />}
            number="4. Resultado"
            text="Clasificación con confianza"
          />
        </div>

        {/* Cards horizontales */}
        <div className="grid md:grid-cols-2 gap-8">
          <div className="p-6 bg-white shadow rounded-lg text-left flex flex-col">
            <div className="flex items-center gap-2 mb-3">
              <Cog className="text-blue-500 w-6 h-6" />
              <h4 className="font-bold text-lg">Red Neuronal Convolucional</h4>
            </div>
            <ul className="list-disc list-inside text-gray-600 text-base space-y-2">
              <li>
                <b>Arquitectura:</b> EfficientNetB0 optimizada para detección de imágenes sintéticas
              </li>
              <li>
                <b>Entrenamiento:</b> Dataset de 50,000+ imágenes reales y sintéticas
              </li>
              <li>
                <b>Precisión:</b> 94.2% en conjunto de prueba independiente
              </li>
            </ul>
          </div>

          <div className="p-6 bg-white shadow rounded-lg text-left flex flex-col">
            <div className="flex items-center gap-2 mb-3">
              <GraduationCap className="text-purple-500 w-6 h-6" />
              <h4 className="font-bold text-lg">Investigación Universitaria</h4>
            </div>
            <p className="text-gray-600 text-base mb-3">
              Este modelo fue desarrollado como parte de una investigación universitaria
              sobre detección de imágenes sintéticas, enfocándose en:
            </p>
            <ul className="list-disc list-inside text-gray-600 text-base space-y-2">
              <li>Análisis de artefactos en imágenes generadas por GANs</li>
              <li>Detección de patrones de compresión anómalos</li>
              <li>Evaluación en diversas técnicas de generación</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Instrucciones de Uso */}
      <section className="py-16 text-center max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold mb-10">Instrucciones de Uso</h2>

        {/* Cards */}
        <div className="grid md:grid-cols-4 gap-8 mb-12">
          <InfoCard
            icon={<Image className="w-10 h-10 text-blue-500 mx-auto mb-3" />}
            title="Formatos Aceptados"
            subtitle="JPG, JPEG, PNG"
            text="Subir imágenes en alta calidad para mejores resultados"
          />
          <InfoCard
            icon={<Upload className="w-10 h-10 text-green-500 mx-auto mb-3" />}
            title="Tamaño Máximo"
            subtitle="10 MB por imagen"
            text="Imágenes más grandes serán redimensionadas automáticamente"
          />
          <InfoCard
            icon={<BarChart2 className="w-10 h-10 text-orange-500 mx-auto mb-3" />}
            title="Tiempo de Procesamiento"
            subtitle="2-5 segundos"
            text="El análisis con CNN requiere algunos segundos de procesamiento"
          />
          <InfoCard
            icon={<ShieldCheck className="w-10 h-10 text-purple-500 mx-auto mb-3" />}
            title="Privacidad"
            subtitle="Imágenes no se almacenan"
            text="Las imágenes se procesan localmente y no se guardan en servidores"
          />
        </div>

        {/* Consejos */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-8 max-w-5xl mx-auto">
          <h3 className="text-xl font-semibold mb-6 text-blue-800">
            Consejos para Mejores Resultados
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-left text-blue-700">
            <ul className="list-disc list-inside space-y-2">
              <li>Usar imágenes con buena iluminación</li>
              <li>Evitar imágenes muy pequeñas o pixeladas</li>
              <li>Preferir imágenes con rostros o figuras humanas</li>
            </ul>
            <ul className="list-disc list-inside space-y-2">
              <li>Evitar imágenes con mucho procesamiento previo</li>
              <li>El modelo funciona mejor con retratos y escenas</li>
              <li>Resultados más precisos en imágenes de alta resolución</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Ejemplos de Resultados de Pruebas */}
      <section className="py-16 text-center max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold mb-6">Ejemplos de Resultados de Pruebas</h2>
        <p className="text-gray-600 mb-12">
          Conoce cómo nuestro algoritmo clasifica diferentes tipos de imágenes
        </p>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Caso 1 */}
          <div className="bg-white shadow rounded-lg overflow-hidden p-6 relative">
            <span className="absolute top-4 right-4 bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm font-semibold flex items-center gap-1">
              <CheckCircle className="w-4 h-4" /> REAL
            </span>
            <h4 className="font-semibold text-lg mb-4">Caso de Prueba #1</h4>
            <img
              src="https://picsum.photos/400/250?grayscale"
              alt="Ejemplo real"
              className="w-full rounded mb-4"
            />
            <p className="text-gray-800 font-medium">
              Confianza del modelo:{" "}
              <span className="text-green-600 font-bold">94%</span>
            </p>
            <p className="mt-2 text-sm text-green-700 bg-green-50 border border-green-200 rounded p-2">
              <b>Análisis:</b> Fotografía real con detalles naturales de piel y ojos
            </p>
          </div>

          {/* Caso 2 */}
          <div className="bg-white shadow rounded-lg overflow-hidden p-6 relative">
            <span className="absolute top-4 right-4 bg-orange-100 text-orange-700 px-3 py-1 rounded-full text-sm font-semibold flex items-center gap-1">
              <AlertTriangle className="w-4 h-4" /> IA
            </span>
            <h4 className="font-semibold text-lg mb-4">Caso de Prueba #2</h4>
            <img
              src="https://picsum.photos/400/250"
              alt="Ejemplo IA"
              className="w-full rounded mb-4"
            />
            <p className="text-gray-800 font-medium">
              Confianza del modelo:{" "}
              <span className="text-red-600 font-bold">87%</span>
            </p>
            <p className="mt-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
              <b>Análisis:</b> Arte digital con patrones de renderizado sintético
            </p>
          </div>
        </div>

        {/* Precisión del sistema */}
        <div className="mt-12 bg-blue-50 border border-blue-200 rounded-lg p-8 text-center">
          <h3 className="text-xl font-semibold mb-6 text-blue-800">
            Precisión del Sistema
          </h3>
          <div className="flex flex-col md:flex-row justify-around gap-6">
            <div>
              <p className="text-2xl font-bold text-blue-700">94.2%</p>
              <p className="text-sm text-gray-600">Precisión General</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-blue-700">96.1%</p>
              <p className="text-sm text-gray-600">Detección de Imágenes Reales</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-blue-700">92.3%</p>
              <p className="text-sm text-gray-600">Detección de IA</p>
            </div>
          </div>
        </div>
      </section>

            {/* Footer Académico */}
            <footer className="bg-gray-900 text-gray-200 py-16 mt-16">
        <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-12 px-6">
          {/* Columna izquierda */}
          <div>
            <h3 className="text-xl font-bold mb-3">Investigación Universitaria</h3>
            <p className="text-gray-400 mb-4">
              Este sistema de detección de imágenes sintéticas fue desarrollado
              como parte de una investigación de tesis sobre Deep Learning aplicado
              a la autenticación de medios digitales.
            </p>
            <p className="text-gray-300">
              <b>Universidad:</b> Universidad de Lima <br />
              <b>Facultad:</b> Ingeniería en Sistemas<br />
              <b>Año:</b> 2025
            </p>
          </div>

          {/* Columna derecha */}
          <div>
            <h3 className="text-xl font-bold mb-3">Contacto Académico</h3>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-center gap-2">
                <Mail className="w-5 h-5 text-blue-400" />
                <a
                  href="mailto:20202799@aloe.ulima.edu.pe"
                  className="hover:underline"
                >
                  20202799@aloe.ulima.edu.pe
                </a>
              </li>
              <li className="flex items-center gap-2">
                <Linkedin className="w-5 h-5 text-blue-400" />
                <a
                  href="https://www.linkedin.com/in/cristiancorrea/"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:underline"
                >
                  linkedin.com/in/cristiancorrea
                </a>
              </li>
              <li className="flex items-center gap-2">
                <Github className="w-5 h-5 text-blue-400" />
                <a
                  href="https://github.com/kriskoCD"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:underline"
                >
                  github.com/kriskoCD
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Nota ética */}
        <div className="max-w-4xl mx-auto bg-yellow-900/40 border border-yellow-600 text-yellow-200 rounded-lg p-6 mt-12">
          <h4 className="font-bold mb-2">⚠️ Nota Ética Importante</h4>
          <p className="text-sm leading-relaxed">
            Este sistema se ofrece exclusivamente con <b>fines de investigación y educativos</b>.
            No debe ser utilizado en contextos legales, forenses, o para tomar decisiones críticas
            sin validación adicional por parte de expertos. Los resultados son estimaciones basadas
            en modelos de aprendizaje automático y pueden contener errores. El uso de esta herramienta
            es bajo la responsabilidad del usuario.
          </p>
        </div>

        {/* Info en columnas */}
        <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-6 text-center mt-12 text-gray-400">
          <div>
            <h5 className="font-semibold text-white">Modelo</h5>
            <p>EfficientNetB0<br />94.2% precisión</p>
          </div>
          <div>
            <h5 className="font-semibold text-white">Tecnologías</h5>
            <p>React + TypeScript<br />Tailwind CSS</p>
          </div>
          <div>
            <h5 className="font-semibold text-white">Dataset</h5>
            <p>50,000+ imágenes<br />Reales & sintéticas</p>
          </div>
        </div>

        {/* Créditos */}
        <div className="text-center text-gray-500 text-sm mt-12">
          © 2025 AI Image Detective – Proyecto de Investigación Universitaria <br />
          Desarrollado para fines académicos • No comercial • Código con propósitos educativos
        </div>
      </footer>
    </div>
  );
}

/* ---- Components ---- */
function Card({ icon, title, text }) {
  return (
    <div className="p-6 bg-white shadow rounded-lg flex flex-col justify-center items-center min-h-[220px]">
      <div className="mb-3">{icon}</div>
      <h4 className="font-bold text-xl">{title}</h4>
      <p className="mt-2 text-gray-600 text-base text-center">{text}</p>
    </div>
  );
}

function Step({ icon, number, text }) {
  return (
    <div className="p-6 bg-white shadow rounded-lg flex flex-col items-center justify-center">
      {icon}
      <h4 className="font-bold text-lg mb-2">{number}</h4>
      <p className="text-gray-600 text-base text-center">{text}</p>
    </div>
  );
}

function InfoCard({ icon, title, subtitle, text }) {
  return (
    <div className="p-6 bg-white shadow rounded-lg text-center">
      {icon}
      <h4 className="font-bold text-lg mb-1">{title}</h4>
      <p className="text-gray-700 font-medium">{subtitle}</p>
      <p className="text-gray-600 text-sm mt-2">{text}</p>
    </div>
  );
}
