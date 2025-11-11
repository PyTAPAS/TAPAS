.. _tut_new_model:

Add a custom kinetic model
==========================

This example is taken from *TAPAS: Transient Absorption Processing & Analysis Software*, ChemRxiv preprint, **2025**, DOI: https://doi.org/10.26434/chemrxiv-2025-4d7fh, CC BY 4.0


First, familiarize yourself with the backend architecture found :ref:`here <software_architecture>`.


Adding a custom kinetic model to TAPAS entails three steps:

#. **Define the kinetics**  

   Edit ``tapas/utils/model_function.py``:  

   1. Duplicate an existing template function inside the ``ModelFunctions`` class.  
   2. Rename it using the pattern ``model_<n>C_<m>k_<j>``, where:  

      - ``n`` is the number of compartments  
      - ``m`` is the number of first-order rate constants  
      - ``j`` is an arbitrary identifier  
      TAPAS parses this name to prefill the rate-constant input masks.  
   3. In the ``# parameters`` section, set your rate constants (``k1, k2, …``) and update ``n_pool = <n>``.  

      .. code-block:: python

        # ---------- parameters --------------------------------------------------------------------
        t0, irf = theta[0],  theta[1]
        k1, k2, k3, k4, k5, k6 = 1.0 / theta[2:-2] if use_bleach else 1 / theta[2:]

        n_pool = 5 if Ainf else 4
        c0 = jnp.zeros(n_pool)



   4. Modify the ``kinetics()`` sub-function to assemble the rate matrix that integrates to concentration profiles.  

      .. code-block:: python

        # ---------- kinetics ----------------------------------------------------------------------
        def kinetics(c, g):
            A = c[0]
            B = c[1]
            C = c[2]
            D = c[3]

            dA = g - k1 * A
            dB = k1 * A - (k2 + k6) * B
            dC = k2 * B - k3 * C
            dD = k3 * C - (k4 + k5) * D

            if Ainf:
                E = c[4]
                dE = k5 * D
                return jnp.array([dA, dB, dC, dD, dE])
            else:
                return jnp.array([dA, dB, dC, dD])



   5. Leave all other code unchanged.

#. **Expose the model in the GUI**  

   Edit ``tapas/views/tabwidgets/global_fit_tabwidget.py``:  

   In the ``InputWidget._initUI()`` method, append your new model name string to ``self.cb_model`` so it appears in the Model combo-box.

    .. code-block:: python

       self.cb_model.addItems(
          ['parallel', 'sequential', '2C_3k_1', '3C_5k_1', '3C_4k_1', '4C_6k_1', ])

#. **(Optional) Label the compartments**  

   Edit ``tapas/controller/global_fit_controller.py``:  

   - In ``GlobalFitController.get_component_labels()``, insert your model identifier and default labels into the returned mapping.  

      .. code-block:: python

        elif model == '4C_6k_1':
            labels += ['A', 'B', 'C', 'D']
            if Ainf:
                labels.append('E_{inf}')   

   - These labels will propagate to plots, tables, and exported CSV files.









