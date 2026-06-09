"""Function-calling / tool-use — registro de tools usadas pelo agente.

Reaproveita o LAB-001. Voce vai preencher 1 TODO aqui (sua tool especifica).
"""

from __future__ import annotations

import json
from typing import Any, Callable


# ============================================================================
# TODO 4 — Sua tool especifica do dominio
# ============================================================================
# Cada projeto precisa de UMA tool customizada que faca sentido para o problema.
# Exemplos por dominio:
#   - Livro tecnico:    lookup_chapter(chapter: int) -> str
#   - Changelog:        check_compat(lib: str, version: str) -> dict
#   - Podcast:          get_timestamp(quote: str) -> str
#   - Codigo:           run_snippet(code: str) -> str  (sandboxed)
#   - Documentos legais: cite_article(law: str, article: int) -> str
#
# 1. Implemente a funcao Python real abaixo (substitua o exemplo)
# 2. Adicione o schema JSON em TOOLS abaixo
# 3. Registre em TOOL_REGISTRY
# ============================================================================


# SEU CODIGO AQUI — TODO 4
PPC_SECTIONS: dict[str, dict[str, Any]] = {
    "identificacao": {
        "titulo": "1. Identificação da Instituição e do Curso",
        "pagina": 6,
        "descricao": "Nome do curso, campus São Carlos, CNPJ, carga horária total (3207,2h), duração (8 semestres), início em 1º sem/2023.",
    },
    "justificativa": {
        "titulo": "2. Justificativa e Demanda de Mercado",
        "pagina": 14,
        "descricao": "Contexto do mercado de software, demanda regional em São Carlos, polo tecnológico com Embrapa, UFSCar, USP.",
    },
    "acesso": {
        "titulo": "3. Requisitos e Formas de Acesso",
        "pagina": 17,
        "descricao": "Acesso via SISU/ENEM, vagas por semestre, requisito de ensino médio completo.",
    },
    "perfil_egresso": {
        "titulo": "4. Perfil do Egresso",
        "pagina": 17,
        "descricao": "Bacharel que conhece, adapta e atua nas fases do processo de desenvolvimento de software com qualidade, sistematização, controle, eficácia e eficiência.",
    },
    "objetivos": {
        "titulo": "5. Objetivos do Curso",
        "pagina": 19,
        "descricao": "Objetivo geral e específicos: formar profissionais para desenvolvimento, gerência, qualidade e inovação em software.",
    },
    "curriculo": {
        "titulo": "6. Organização Curricular",
        "pagina": 21,
        "descricao": "Estrutura dos 8 semestres com 47 disciplinas obrigatórias + LIBRAS optativa. 1º sem: Algoritmos 1, Processos de Produção de Software, Intro à Web, Arquitetura de Computadores, Matemática para Computação, Inglês 1.",
    },
    "prerequisitos": {
        "titulo": "6.4. Pré-requisitos",
        "pagina": 25,
        "descricao": "Tabela de pré-requisitos encadeados, ex: Algoritmos 2 requer Algoritmos 1; Estrutura de Dados requer Algoritmos 2.",
    },
    "estagio": {
        "titulo": "6.5. Estágio Curricular Supervisionado",
        "pagina": 26,
        "descricao": "Estágio não obrigatório. Pode ser convertido em Atividades Complementares. Carga horária mínima de 280h por estágio.",
    },
    "atividades_complementares": {
        "titulo": "6.7. Atividades Complementares",
        "pagina": 30,
        "descricao": "640 horas de ACs obrigatórias. Incluem eventos, cursos, publicações, monitoria, estágio, participação em projetos.",
    },
    "extensao": {
        "titulo": "10. Atividades de Extensão",
        "pagina": 43,
        "descricao": "4 disciplinas de Atividades de Extensão (AXB1–4) nos semestres 4, 6, 7 e 8. Mínimo de 10% da carga horária total.",
    },
    "metodologia": {
        "titulo": "7. Metodologia",
        "pagina": 36,
        "descricao": "Metodologias ativas, PBL, aprendizagem colaborativa, projetos integradores. Uso de laboratórios de informática.",
    },
    "avaliacao": {
        "titulo": "8. Avaliação da Aprendizagem",
        "pagina": 39,
        "descricao": "Nota mínima 6,0 para aprovação. Avaliações formativas e somativas. Recuperação paralela disponível.",
    },
    "planos_ensino": {
        "titulo": "18. Planos de Ensino",
        "pagina": 62,
        "descricao": "Ementas, objetivos, conteúdo programático e bibliografia de todas as 47 disciplinas do curso.",
    },
    "infraestrutura": {
        "titulo": "17. Infraestrutura",
        "pagina": 59,
        "descricao": "Laboratórios de informática, biblioteca, salas de aula. Acessibilidade. Campus São Carlos do IFSP.",
    },
}


def lookup_section(section_key: str) -> str:
    """Retorna informações sobre uma seção do PPC pelo nome da seção."""
    if section_key in PPC_SECTIONS:
        s = PPC_SECTIONS[section_key]
        return f"{s['titulo']} (p. {s['pagina']}): {s['descricao']}"
    keys = ", ".join(PPC_SECTIONS.keys())
    return f"Seção '{section_key}' não encontrada. Seções disponíveis: {keys}"


TOOLS: list[dict[str, Any]] = [
    # SEU CODIGO AQUI — TODO 4 (continuacao)
    {
        "type": "function",
        "function": {
            "name": "lookup_section",
            "description": "Retorna informações estruturadas sobre uma seção específica do PPC (Projeto Pedagógico de Curso) de Engenharia de Software do IFSP São Carlos. Use quando o usuário perguntar sobre uma parte específica do documento, como currículo, TCC, estágio, perfil do egresso, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_key": {
                        "type": "string",
                        "description": "Chave da seção. Valores válidos: identificacao, justificativa, acesso, perfil_egresso, objetivos, curriculo, prerequisitos, estagio, atividades_complementares, extensao, metodologia, avaliacao, planos_ensino, infraestrutura",
                    }
                },
                "required": ["section_key"],
            },
        },
    },
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "lookup_section": lookup_section,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
