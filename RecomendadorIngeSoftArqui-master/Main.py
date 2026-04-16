import os

def menu():
    while True:
        print("\n" + "="*50)
        print(" SISTEMA ACADÉMICO ")
        print("="*50)
        print("1. 📊 Ejecutar análisis completo")
        print("2. 🎯 Ejecutar recomendador")
        print("3. ❌ Salir")
        print("="*50)

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("\nEjecutando análisis...\n")
            os.system("python ModeloEstadisticoIAHibrido.py")

        elif opcion == "2":
            print("\nEjecutando recomendador...\n")
            os.system("python recomendador.py")

        elif opcion == "3":
            print("Saliendo del sistema...")
            break

        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    menu()
