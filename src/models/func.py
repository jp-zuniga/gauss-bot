"""
Implementación de clase representando funciones matemáticas.
"""

from decimal import Decimal
from logging import WARNING, getLogger
from re import compile as comp, sub

from PIL.Image import open as open_img
from customtkinter import CTkImage
from matplotlib import use
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.pyplot import axis, close, rc, savefig, subplots, text
from sympy import (
    Expr,
    Interval,
    Reals,
    Symbol,
    diff,
    integrate,
    latex,
    nan,
    oo,
    parse_expr,
    zoo,
)
from sympy.calculus.util import continuous_domain
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)

from src.utils import SAVED_FUNCS_PATH, transparent_invert

# use("TkAgg")
getLogger("matplotlib").setLevel(WARNING)

TRANSFORMS: tuple = (*standard_transformations, implicit_multiplication_application)


class Func:
    """
    Representa una función matemática.
    """

    def __init__(self, nombre: str, expr: str, latexified: bool = False) -> None:
        """
        Args:
            nombre:     Nombre de la función en la notación f(x).
            expr:       La expresión matemática que define la función.
            latexified: Si la función se ha convertido en notación LaTeX.

        Raises:
            ValueError: Si la expresión tiene un dominio complejo.

        """

        self.nombre = nombre

        var_pattern: str = r"\(([a-z])\)"
        self.var = Symbol(comp(var_pattern).findall(nombre)[0])

        self.latexified = latexified
        self.latex_img: CTkImage | None = None

        def replace_var(expr: str) -> str:
            var_str = str(self.var)
            patt: str = r"\bx\b"

            replaced_expr = expr
            for _ in list(comp(patt).finditer(expr)):
                replaced_expr = sub(patt, var_str, expr)
            return replaced_expr

        if "sen" in expr:
            expr = expr.replace("sen", "sin")
        if "^" in expr:
            expr = expr.replace("^", "**")
        if "e**" in expr:
            expr = expr.replace("e**", "exp")

        expr = replace_var(expr)
        self.expr: Expr = parse_expr(expr, transformations=TRANSFORMS)

        if self.expr.has(oo, -oo, zoo, nan):
            raise ValueError("La función tiene un dominio complejo.")

    def __str__(self) -> str:
        """
        Crear ecuación matemática con 'self.nombre' y 'self.expr'.
        """

        return f"{self.nombre} = {self.expr}"

    def get_dominio(self) -> Interval:
        """
        Wrapper para continuous_domain() de sympy.

        Returns:
            Interval: El dominio real de 'self.expr'.

        """

        return continuous_domain(self.expr, self.var, Reals)

    def es_continua(self, intervalo: tuple[Decimal, Decimal]) -> bool:
        """
        Valida si 'self.expr' es continua en el
        intervalo indicado, con respecto a 'self.var'.

        Args:
            intervalo: Par de números que encierran el intervalo.

        Returns:
            bool: Si la función es continua o no.

        """

        a, b = intervalo
        return Interval(a, b, left_open=True, right_open=True).is_subset(
            self.get_dominio(),
        )  # type: ignore[reportReturnType]

    def derivar(self) -> "Func":
        """
        Encontrar la derivada de self.

        Returns:
            Func: Derivada de 'self.expr'.

        """

        if "'" not in self.nombre:
            d_nombre = f"{self.nombre[0]}'{self.nombre[1:]}"

        else:
            num_diff: int = self.nombre.count("'")
            d_nombre: str = self.nombre.replace(
                "".join("'" for _ in range(num_diff)),
                "".join("'" for _ in range(num_diff + 1)),
            )

        return Func(d_nombre, str(diff(self.expr, self.var)))

    def integrar(self) -> "Func":
        """
        Encontrar la integral indefinida de self.

        Returns:
            Func: Integral indefinida de 'self.expr'.

        """

        try:
            match = list(comp(r"\^\(-\d+\)").finditer(self.nombre))[-1]
        except IndexError:
            match = None

        if match is not None:
            num_ints = int(next(x for x in self.nombre if x.isdigit()))
            i_nombre: str = (
                self.nombre[: match.start()]
                + f"^(-{num_ints + 1})"
                + self.nombre[match.end() :]
            )
        else:
            i_nombre = rf"{self.nombre[0]}^{'(-1)'}{self.nombre[1:]}"

        return Func(i_nombre, str(integrate(self.expr, self.var)))

    def get_di_nombre(self, diffr: bool = False, integ: bool = False) -> str:
        """
        Encontrar el nombre de la derivada o integral de self.

        Args:
            diffr: Si se quiere encontrar el nombre de la derivada.
            integ: Si se quiere encontrar el nombre de la integral.

        Returns:
            str: El nombre de la derivada/integral.

        Raises:
            NotImplementedError: Bajo circumstancias misteriosas inimaginables.

        """

        if diffr:
            return self.get_derivada_nombre(self.nombre.count("'"))
        if integ:
            return self.get_integral_nombre(
                int(next((x for x in self.nombre if x.isdigit()), 0)),
            )
        raise NotImplementedError("¿Cómo llegamos aquí?")

    def get_derivada_nombre(self, num_diff: int) -> str:
        """
        Formatear el nombre de la derivada en la
        notación dy/dx en LaTeX, con respecto a la variable.

        Args:
            num_diff: Cuantas veces se ha derivado la función original.

        Returns:
            str: El nombre de la derivada formateado.

        """

        return (
            f"\\frac{{d{f'^{{{num_diff}}}' if num_diff > 1 else ''}{self.nombre[0]}}}"
            f"{{d{self.var!s}{f'^{{{num_diff}}}' if num_diff > 1 else ''}}}"
        )

    def get_integral_nombre(self, num_integ: int) -> str:
        """
        Formatear el nombre de la integral con la
        notación ∫ f(x) dx, con respecto a la variable.

        Args:
            num_integ: Número de veces que se ha integrado la función original.

        Returns:
            str: El nombre del integral formateado.

        """

        return (
            rf"{r''.join(r'\int' for _ in range(num_integ))}"
            rf" {self.nombre[0] + rf'({self.var})'}\, d{self.var!s}"
        )

    def get_png(self) -> CTkImage:
        """
        Retornar una imagen PNG de self.

        Returns:
            CTkImage: Imagen de self generada con LaTeX y matplotlib.

        """

        if not self.latexified or self.latex_img is None:
            self.latex_img = Func.latex_to_png(
                nombre_expr=(
                    self.nombre
                    if (self.nombre.count("'") == 0 and self.nombre.count("∫") == 0)
                    else self.get_di_nombre()
                ),
                expr=str(self.expr),
                con_nombre=True,
            )

            self.latexified = True
        return self.latex_img

    @staticmethod
    def latex_to_png(
        file_name: str | None = None,
        nombre_expr: str | None = None,
        expr: str | None = None,
        misc_str: str | None = None,
        con_nombre: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> CTkImage:
        """
        Convertir texto a formato LaTeX para crear una imagen PNG.

        Args:
            file_name:   Nombre del archivo de salida.
            nombre_expr: Nombre de la función a mostrar.
            expr:        String de la expresión a convertir.
            misc_str:    String de texto a convertir.
            con_nombre:  Si se debe incluir el nombre de la función en la imagen.
            kwargs:      Kwargs para sympy.latex().

        Returns:
            CTkImage: Imagen PNG creada a partir del texto LaTeX generado.

        Raises:
            ValueError: Si no se recibe texto a convertir.

        """

        SAVED_FUNCS_PATH.mkdir(exist_ok=True)
        output_file = SAVED_FUNCS_PATH / rf"{file_name or nombre_expr}.png"

        if expr is not None:
            latex_str: str = latex(
                parse_expr(expr, transformations=TRANSFORMS),
                ln_notation=kwargs.pop("ln_notation", True),
            )
        elif misc_str is not None:
            latex_str = misc_str.replace("log", "ln")
        else:
            raise ValueError("No se recibió un string a convertir en PNG.")

        if con_nombre and nombre_expr is not None:
            latex_str = rf"{nombre_expr} = {latex_str}"

        rc("text", usetex=True)
        rc("font", family="serif")

        temp_fig, temp_ax = subplots()
        temp_text = temp_ax.text(
            0.5,
            0.5,
            f"${latex_str}$",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=kwargs.get("font_size", 100),
            transform=temp_ax.transAxes,
        )

        temp_canvas = FigureCanvasAgg(temp_fig)
        temp_canvas.draw()

        bbox = temp_text.get_window_extent(renderer=temp_canvas.get_renderer())
        width, height = (
            (bbox.width / temp_fig.dpi) + 2,
            (bbox.height / temp_fig.dpi) + 1,
        )

        close(temp_fig)

        fig, _ = subplots(figsize=(width, height))
        axis("off")
        text(
            0.5,
            0.5,
            f"${latex_str}$",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=kwargs.pop("font_size", 100),
        )

        savefig(output_file, format="png", transparent=True, dpi=200, pad_inches=0.1)

        close(fig)

        img = open_img(output_file)
        inverted_img = transparent_invert(img)
        return CTkImage(
            dark_image=inverted_img,
            light_image=img,
            size=(int(width * 20), int(height * 20)),
        )
