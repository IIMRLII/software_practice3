package com.example.myapplication

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.navigation.fragment.findNavController
import com.example.myapplication.databinding.FragmentSecondBinding
import kotlin.math.floor

/**
 * A simple [Fragment] subclass as the second destination in the navigation.
 */
class SecondFragment : Fragment() {

    private var _binding: FragmentSecondBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
            inflater: LayoutInflater, container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {

        _binding = FragmentSecondBinding.inflate(inflater, container, false)
        return binding.root

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.buttonSecond.setOnClickListener {
            findNavController().navigate(R.id.action_SecondFragment_to_FirstFragment)
        }

        val showTextView = view.findViewById<TextView>(R.id.textview_word)
//        val count = arguments?.getString("int")?.toInt()
        val count = arguments?.let { FirstFragmentArgs.fromBundle(it).num }
        showTextView.text = "random number between " + count.toString() + " and 0";
        val showCountTextView = view.findViewById<TextView>(R.id.textview_first)
        showCountTextView.text = floor(Math.random() * count!!).toInt().toString()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}