    def _apply_validated_modifications(self):
        """Aplica modificações validadas com canary deployment."""
        from penin_self_modification_canary import apply_canary_modification  # type: ignore

        for modification_id, modification in list(self.active_modifications.items()):
            if modification.validation_status == ValidationStatus.VALIDATED:
                try:
                    # Usa canary deployment
                    success = apply_canary_modification({
                        "target_file": modification.target_file,
                        "old_code": modification.old_code,
                        "new_code": modification.new_code
                    })

                    if success:
                        modification.validation_status = ValidationStatus.VALIDATED
                        self.completed_modifications.append(modification)
                        self.modification_state.successful_modifications += 1
                        logger.info(f"✅ Modificação canary aplicada com sucesso: {modification_id}")
                    else:
                        modification.validation_status = ValidationStatus.FAILED
                        self.modification_state.failed_modifications += 1
                        logger.warning(f"⚠️ Modificação canary falhou/rollback: {modification_id}")

                    # Remove da lista ativa
                    del self.active_modifications[modification_id]

                except Exception as e:
                    logger.error(f"Erro no canary deployment {modification_id}: {e}")
                    self.modification_state.failed_modifications += 1
                    del self.active_modifications[modification_id]