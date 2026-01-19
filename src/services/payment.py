def calcular_pago(tipo: str, consumo: float, electricity_data=None, water_data=None):

    if tipo == "luz":
        consumo_kwh = consumo

        region = electricity_data.get("region", "Costa")
        adulto_mayor = electricity_data.get("adulto_mayor", "No")
        discapacidad = electricity_data.get("discapacidad", "No")
        incluye_tasas = electricity_data.get("incluye_tasas", "No")

        LIMITE_DIGNIDAD = 130 if region == "Costa" else 110
        TARIFA_DIGNIDAD = 0.045
        TARIFA_NORMAL = 0.0805  
        CARGO_FIJO = 1.41

        if consumo_kwh <= LIMITE_DIGNIDAD:
            costo_consumo = consumo_kwh * TARIFA_DIGNIDAD
        else:
            costo_consumo = consumo_kwh * TARIFA_NORMAL

        descuento = 0.0
        if adulto_mayor == "Si" or discapacidad == "Si":
            descuento = 0.50 * costo_consumo

        subtotal = costo_consumo - descuento + CARGO_FIJO

        tasas = 0.0
        if incluye_tasas == "Si":
            tasas = 1.77  

        total = subtotal + tasas

        return round(total, 2)

    if tipo != "agua":
        return None

    C = consumo
    agua = 0.0

    if C > 0:
        tramo = min(C, 15)
        agua += tramo * 0.352
        C -= tramo

    if C > 0:
        tramo = min(C, 15)
        agua += tramo * 0.521
        C -= tramo

    if C > 0:
        tramo = min(C, 30)
        agua += tramo * 0.738

    alcantarillado = 0.0
    if water_data.get("alcantarillado_sanitario") == "Si":
        alcantarillado = 0.80 * agua

    cargo_fijo = 1.43
    subtotal = agua + alcantarillado + cargo_fijo

    descuento = 0.0
    if water_data.get("personas_mayores_discapacitadas") == "Si":
        descuento = 0.50 * (agua + alcantarillado)

    otros = 0.0

    total = subtotal - descuento + otros
    return round(total, 2)
