# Checklist QA Endpoint Multimodal

## 1. Verificación básica

- [ ] Endpoint /predict/multimodal está activo
- [ ] Devuelve mensaje de inicio en GET /

## 2. Inputs válidos

- [ ] Datos cuantitativos correctos + imagen correcta
- [ ] Predicción final entregada (text_prediction, image_prediction, final_prediction)
- [ ] Probabilidades devueltas entre 0 y 1

## 3. Inputs inválidos

- [ ] Datos faltantes
- [ ] Imagen faltante
- [ ] Tipos de datos incorrectos (string en lugar de int, float donde no corresponde)
- [ ] Valores fuera de rango (0 o >10 si aplica según dataset)
- [ ] Imagen corrupta (no legible)
- [ ] Imagen demasiado grande (verificar que se redimensione a 224x224)
