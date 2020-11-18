class MalformedGeneLabelException(ValueError):
    def __init__(self, gene_label: str):
        self.message = f"{gene_label} is not in the format GENE_SYMBOL (ENTREZ_ID)"
        self.gene_label = gene_label
